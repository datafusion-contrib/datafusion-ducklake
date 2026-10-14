//! SQLite implementation of [`MetadataWriter`].
//!
//! Requires multi-threaded Tokio runtime (`#[tokio::test(flavor = "multi_thread")]`).

use std::{
    fs::{File, OpenOptions},
    path::PathBuf,
    str::FromStr,
    sync::{Arc, Mutex},
};

use crate::Result;
use crate::error::{TypeChangeOperation, TypeChangeWriteMode};
use crate::inlined_reclaim::{SUPERSEDED_INLINED_TABLES_SQL, dead_table_inlined_storage};
use crate::maintenance::{
    CleanupCriteria, ExpireCriteria, ExpiredSnapshot, ScheduledFile, format_sql_timestamp,
};
use crate::metadata_provider::{InlinedSystemColumns, TagObjectType, TagTarget, block_on};
use crate::metadata_provider_sqlite::{catalog_inlined_system_columns, inlined_table_layout};
use crate::metadata_writer::{
    ColumnDef, ColumnStat, CommitIds, ConflictScope, DataFileInfo, DeleteFileEntry, DeleteFileInfo,
    ExistingCatalogColumn, INLINED_INDEX_COLUMNS_SETTING, InlinedRowRef, MetadataWriter,
    MultiTableCommit, SnapshotCommitMetadata, StagedTableData, StagedTableWrite, WriteMode,
    WriteSetupResult, assign_column_ids, catalog_column_defs, catalog_column_type_equal,
    catalog_column_type_requires_migration, catalog_columns_differ, encode_inlined_index_columns,
    inlined_delete_groups, live_inlined_index_columns, parse_inlined_index_columns,
    snapshot_has_change, staged_table_write_changes, table_storage_changes, table_write_changes,
    top_level_column_ids, validate_delete_entries, validate_inlined_index_columns, validate_name,
    validate_table_setting,
};
use crate::metadata_writer::{
    DroppedDataFile, InlinedFileDelete, InlinedFlushFileDelete, SQL_GLOBAL_INLINING_COLUMNS,
    SQL_TABLE_INLINING_COLUMNS, SourceFile, TableChanges, enables_inlining,
    inlined_deletion_conflict, reject_reserved_inlining_columns, rewritten_delete_conflict,
    rewritten_deletes_snapshot, row_mutation_source_files, source_file_id_list, staged_conflict,
    table_conflict, validate_row_mutation_stage,
};
use crate::metadata_writer::{
    InlinedFlushDeleteFile, InlinedFlushTable, carried_row_id_start, inlined_flush_changes,
    inlined_flush_conflict, validate_inlined_flush,
};
use crate::metadata_writer::{directory_path, tag_change};
use crate::partition::PartitionTransform;
use arrow::array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, FixedSizeBinaryArray, Float32Array,
    Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, LargeBinaryArray,
    LargeStringArray, StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use sqlx::AssertSqlSafe;
use sqlx::QueryBuilder;
use sqlx::Row;
use sqlx::sqlite::{Sqlite, SqliteConnectOptions, SqlitePool, SqlitePoolOptions};

const DEFAULT_MAX_CONNECTIONS: u32 = 5;

fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

fn inlined_sqlite_type(data_type: &DataType) -> &'static str {
    match data_type {
        DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32 => "BIGINT",
        // Official declares `VARCHAR` and stores the decimal text.
        DataType::UInt64 => "VARCHAR",
        // DuckDB's text of the value, as official's SQLite catalogs store it:
        // SQLite stores a bound NaN in a REAL column as NULL.
        DataType::Float32 | DataType::Float64 => "VARCHAR",
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView => "BLOB",
        DataType::FixedSizeBinary(size) if *size != 16 => "BLOB",
        _ => "VARCHAR",
    }
}

/// DuckDB's text of an inlined float (`1.5`, `nan`, `-inf`, `-0.0`), which
/// official's SQLite catalogs store in a `VARCHAR` column.
fn inlined_float_text(value: &datafusion::common::ScalarValue) -> String {
    crate::stats_encode::inline_stat_text(value, "double").expect("every float has DuckDB text")
}

/// Whether a declared SQLite column type is a float type an inlined table of
/// an earlier release declared, or another writer's float spelling.
fn is_float_declaration(declared: &str) -> bool {
    matches!(
        declared.trim().to_ascii_uppercase().as_str(),
        "DOUBLE" | "FLOAT" | "REAL" | "DOUBLE PRECISION" | "FLOAT4" | "FLOAT8"
    )
}

/// Rebuild an inlined data table that releases of this crate created with
/// float (`DOUBLE`, `FLOAT`) columns so those columns hold DuckDB's text in
/// `VARCHAR`, as official's do, before more rows are written to it: SQLite
/// stores a NaN bound to a `REAL` column as NULL, and DuckDB's reader rejects
/// text in one. Each stored value keeps its double: it is read into Rust and
/// written as the shortest text that parses back to it, as inserts write it
/// ([`inlined_float_text`]); SQLite's own float-to-text conversions do not
/// round-trip every double.
///
/// The new table is declared from this crate's own type mapping for `fields`
/// (the table's data columns, after its three system columns), never from the
/// declared types or index SQL stored in the catalog, and every identifier is
/// quoted; the caller recreates the declared indexes.
async fn migrate_double_inlined_columns(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    physical_name: &str,
    fields: &[(&str, &DataType)],
) -> Result<()> {
    let columns: Vec<(String, String)> =
        sqlx::query_as("SELECT name, type FROM pragma_table_info(?) ORDER BY cid")
            .bind(physical_name)
            .fetch_all(&mut **tx)
            .await?;
    if !columns
        .iter()
        .skip(3)
        .any(|(_, declared)| is_float_declaration(declared))
    {
        return Ok(());
    }
    let data_columns = &columns[3.min(columns.len())..];
    if data_columns.len() != fields.len()
        || data_columns
            .iter()
            .zip(fields)
            .any(|((name, _), (field, _))| name != field)
    {
        return Err(crate::DuckLakeError::InvalidConfig(format!(
            "inlined data table {physical_name} does not hold the columns of its schema \
             version; it cannot be converted to VARCHAR float columns"
        )));
    }
    let rebuilt = format!("{physical_name}__varchar_floats");
    let definitions = columns
        .iter()
        .take(3)
        .map(|(name, _)| format!("{} BIGINT", quote_ident(name)))
        .chain(fields.iter().map(|(name, data_type)| {
            format!("{} {}", quote_ident(name), inlined_sqlite_type(data_type))
        }))
        .collect::<Vec<_>>()
        .join(", ");
    // Rows are matched by their rowid, under an alias no column shadows.
    let Some(rowid) = ["rowid", "_rowid_", "oid"].into_iter().find(|alias| {
        !columns
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case(alias))
    }) else {
        return Err(crate::DuckLakeError::InvalidConfig(format!(
            "inlined data table {physical_name} has columns named rowid, _rowid_ and oid; \
             its float columns cannot be converted to VARCHAR"
        )));
    };
    // Numbers in float columns are copied as NULL and written below; text (an
    // official writer's, or NULL) is copied as it is.
    let names = columns
        .iter()
        .map(|(name, _)| quote_ident(name))
        .collect::<Vec<_>>()
        .join(", ");
    let values = columns
        .iter()
        .map(|(name, declared)| {
            let column = quote_ident(name);
            if is_float_declaration(declared) {
                format!(
                    "CASE WHEN typeof({column}) IN ('real', 'integer') THEN NULL ELSE {column} END"
                )
            } else {
                column
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    for statement in [
        format!("DROP TABLE IF EXISTS {}", quote_ident(&rebuilt)),
        format!("CREATE TABLE {} ({definitions})", quote_ident(&rebuilt)),
        format!(
            "INSERT INTO {} ({rowid}, {names}) SELECT {rowid}, {values} FROM {}",
            quote_ident(&rebuilt),
            quote_ident(physical_name)
        ),
    ] {
        sqlx::query(AssertSqlSafe(statement))
            .execute(&mut **tx)
            .await?;
    }
    for ((name, declared), (_, data_type)) in columns.iter().skip(3).zip(fields) {
        if !is_float_declaration(declared) {
            continue;
        }
        let column = quote_ident(name);
        let numbers: Vec<(i64, f64)> = sqlx::query_as(AssertSqlSafe(format!(
            "SELECT {rowid}, CAST({column} AS REAL) FROM {} \
             WHERE typeof({column}) IN ('real', 'integer')",
            quote_ident(physical_name)
        )))
        .fetch_all(&mut **tx)
        .await?;
        let update = format!(
            "UPDATE {} SET {column} = ? WHERE {rowid} = ?",
            quote_ident(&rebuilt)
        );
        for (row, value) in numbers {
            let text = match data_type {
                DataType::Float32 => inlined_float_text(&datafusion::common::ScalarValue::Float32(
                    Some(value as f32),
                )),
                _ => inlined_float_text(&datafusion::common::ScalarValue::Float64(Some(value))),
            };
            sqlx::query(AssertSqlSafe(update.clone()))
                .bind(text)
                .bind(row)
                .execute(&mut **tx)
                .await?;
        }
    }
    for statement in [
        format!("DROP TABLE {}", quote_ident(physical_name)),
        format!(
            "ALTER TABLE {} RENAME TO {}",
            quote_ident(&rebuilt),
            quote_ident(physical_name)
        ),
    ] {
        sqlx::query(AssertSqlSafe(statement))
            .execute(&mut **tx)
            .await?;
    }
    Ok(())
}

// Nested values use reference DuckDB literal text so the shared parser and external readers
// agree.
fn sqlite_type_inlines(data_type: &DataType) -> bool {
    crate::nested_inline::type_supports_inlining(data_type)
}

fn push_inlined_sqlite_value(
    query: &mut QueryBuilder<Sqlite>,
    array: &dyn Array,
    row: usize,
) -> Result<()> {
    if array.is_null(row) {
        query.push_bind(Option::<String>::None);
        return Ok(());
    }

    macro_rules! signed {
        ($array:ty) => {{
            query.push_bind(
                array
                    .as_any()
                    .downcast_ref::<$array>()
                    .expect("Arrow data type and array implementation agree")
                    .value(row) as i64,
            );
        }};
    }
    macro_rules! unsigned {
        ($array:ty) => {{
            let value = array
                .as_any()
                .downcast_ref::<$array>()
                .expect("Arrow data type and array implementation agree")
                .value(row);
            query.push_bind(i64::try_from(value).map_err(|_| {
                crate::error::DuckLakeError::Unsupported(format!(
                    "SQLite inlined data cannot represent unsigned value {value}"
                ))
            })?);
        }};
    }

    match array.data_type() {
        DataType::Boolean => {
            query.push_bind(
                array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .expect("Arrow data type and array implementation agree")
                    .value(row),
            );
        },
        DataType::Int8 => signed!(Int8Array),
        DataType::Int16 => signed!(Int16Array),
        DataType::Int32 => signed!(Int32Array),
        DataType::Int64 => signed!(Int64Array),
        DataType::UInt8 => unsigned!(UInt8Array),
        DataType::UInt16 => unsigned!(UInt16Array),
        DataType::UInt32 => unsigned!(UInt32Array),
        DataType::UInt64 => {
            query.push_bind(format!(
                "{}",
                array
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .expect("Arrow data type and array implementation agree")
                    .value(row),
            ));
        },
        DataType::Float32 => {
            let value = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("Arrow data type and array implementation agree")
                .value(row);
            query.push_bind(inlined_float_text(
                &datafusion::common::ScalarValue::Float32(Some(value)),
            ));
        },
        DataType::Float64 => {
            let value = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("Arrow data type and array implementation agree")
                .value(row);
            query.push_bind(inlined_float_text(
                &datafusion::common::ScalarValue::Float64(Some(value)),
            ));
        },
        DataType::Utf8 => {
            query.push_bind(
                array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("Arrow data type and array implementation agree")
                    .value(row)
                    .to_owned(),
            );
        },
        DataType::LargeUtf8 => {
            query.push_bind(
                array
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .expect("Arrow data type and array implementation agree")
                    .value(row)
                    .to_owned(),
            );
        },
        DataType::Binary => {
            query.push_bind(
                array
                    .as_any()
                    .downcast_ref::<BinaryArray>()
                    .expect("Arrow data type and array implementation agree")
                    .value(row)
                    .to_vec(),
            );
        },
        DataType::LargeBinary => {
            query.push_bind(
                array
                    .as_any()
                    .downcast_ref::<LargeBinaryArray>()
                    .expect("Arrow data type and array implementation agree")
                    .value(row)
                    .to_vec(),
            );
        },
        DataType::BinaryView => {
            query.push_bind(
                array
                    .as_any()
                    .downcast_ref::<BinaryViewArray>()
                    .expect("Arrow data type and array implementation agree")
                    .value(row)
                    .to_vec(),
            );
        },
        DataType::FixedSizeBinary(size) if *size != 16 => {
            query.push_bind(
                array
                    .as_any()
                    .downcast_ref::<FixedSizeBinaryArray>()
                    .expect("Arrow data type and array implementation agree")
                    .value(row)
                    .to_vec(),
            );
        },
        _ => {
            query.push_bind(crate::metadata_writer::inlined_text_value(array, row)?);
        },
    };
    Ok(())
}

async fn sqlite_inlined_index_columns(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
) -> Result<Vec<String>> {
    let value: Option<String> = sqlx::query_scalar(
        "SELECT value FROM ducklake_metadata
         WHERE key = ? AND scope = 'table' AND scope_id = ?",
    )
    .bind(INLINED_INDEX_COLUMNS_SETTING)
    .bind(table_id)
    .fetch_optional(&mut **tx)
    .await?;
    let declared = parse_inlined_index_columns(value.as_deref().unwrap_or_default())?;
    let available = sqlx::query_scalar::<_, String>(
        "SELECT column_name FROM ducklake_column
         WHERE table_id = ? AND end_snapshot IS NULL AND parent_column IS NULL",
    )
    .bind(table_id)
    .fetch_all(&mut **tx)
    .await?
    .into_iter()
    .collect::<std::collections::HashSet<_>>();
    Ok(live_inlined_index_columns(declared, &available))
}

async fn ensure_sqlite_physical_indexes(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
    physical_name: &str,
) -> Result<()> {
    let declared = sqlite_inlined_index_columns(tx, table_id).await?;
    let present = sqlx::query_scalar::<_, String>("SELECT name FROM pragma_table_info(?)")
        .bind(physical_name)
        .fetch_all(&mut **tx)
        .await?
        .into_iter()
        .collect::<std::collections::HashSet<_>>();
    for column in declared.iter().map(String::as_str) {
        if !present.contains(column) {
            continue;
        }
        let index_name = format!("{physical_name}_{column}_idx");
        let exists: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM sqlite_master
             WHERE type = 'index' AND name = ? AND tbl_name = ?",
        )
        .bind(&index_name)
        .bind(physical_name)
        .fetch_one(&mut **tx)
        .await?;
        if exists {
            continue;
        }
        sqlx::query(AssertSqlSafe(format!(
            "CREATE INDEX IF NOT EXISTS {} ON {}({})",
            quote_ident(&index_name),
            quote_ident(physical_name),
            quote_ident(column),
        )))
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

/// Render a slice of ids as a SQL `IN (...)` body. Safe to interpolate because
/// the values are `i64` (no injection surface) — same approach the upstream
/// DuckLake C++ takes, and the only option since SQLite lacks array binds.
fn id_list(ids: &[i64]) -> String {
    ids.iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

const SQL_CREATE_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS ducklake_metadata (
    key VARCHAR NOT NULL,
    value VARCHAR NOT NULL,
    scope VARCHAR,
    scope_id INTEGER
);

-- `schema_version` is the per-catalog monotonic schema counter from the DuckLake
-- spec (`ducklake_metadata_manager.cpp:232`): it bumps on a schema change (DDL)
-- and is carried forward unchanged on a data write, exactly mirroring upstream's
-- `if (SchemaChangesMade()) schema_version++` (`ducklake_transaction_state.cpp:1826`)
-- and the validated Postgres writer here. Upstream also stores `next_catalog_id` /
-- `next_file_id` on this row as ITS id allocators; we deliberately omit them (as the
-- Postgres writer does) because this library allocates ids from its own counters
-- (the `next_column_id` metadata row + autoincrement PKs), never from the snapshot.
CREATE TABLE IF NOT EXISTS ducklake_snapshot (
    snapshot_id INTEGER PRIMARY KEY,
    snapshot_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    schema_version INTEGER NOT NULL DEFAULT 0
);

-- Per-table schema-change ledger (DuckLake spec, `ducklake_metadata_manager.cpp:281`):
-- one row per (table that changed schema, snapshot at which it changed). Written on
-- every DDL that leaves the table live (create, column add/remove/reorder, type
-- promotion); NOT written for a drop (the table has no live schema afterward). Rows
-- are never tombstoned (no `end_snapshot`); they go unreferenced once the table is
-- fully expired and are reclaimed by vacuum — same as upstream `DropTables`.
CREATE TABLE IF NOT EXISTS ducklake_schema_versions (
    begin_snapshot INTEGER NOT NULL,
    schema_version INTEGER NOT NULL,
    table_id INTEGER NOT NULL,
    UNIQUE (table_id, begin_snapshot)
);

CREATE TABLE IF NOT EXISTS ducklake_schema (
    schema_id INTEGER PRIMARY KEY,
    schema_name VARCHAR NOT NULL,
    path VARCHAR NOT NULL DEFAULT '',
    path_is_relative BOOLEAN NOT NULL DEFAULT 1,
    begin_snapshot INTEGER NOT NULL,
    end_snapshot INTEGER
);

CREATE TABLE IF NOT EXISTS ducklake_table (
    table_id INTEGER PRIMARY KEY,
    schema_id INTEGER NOT NULL,
    table_name VARCHAR NOT NULL,
    path VARCHAR NOT NULL DEFAULT '',
    path_is_relative BOOLEAN NOT NULL DEFAULT 1,
    begin_snapshot INTEGER NOT NULL,
    end_snapshot INTEGER
);

-- Faithful match of upstream `duckdb/ducklake` `ducklake_column`
-- (`ducklake_metadata_manager.cpp:258`): a BARE table — no PRIMARY KEY, no
-- NOT NULL, no DEFAULT — in upstream's exact column order. `column_id` is
-- deliberately NOT a single-row PK: a column is versioned by
-- `[begin_snapshot, end_snapshot)` and a stable-id type promotion writes a
-- SECOND row with the same `column_id`, which a single-row PK would forbid.
-- One-live-row / non-null / ownership invariants are enforced in the writer
-- code + tests, exactly as upstream guarantees them (its table has no such
-- constraints). The four `*default*` columns + `parent_column` are projected by
-- DuckDB when it reads catalogs we produce; we leave them NULL (no nested-type
-- or column-default writes yet). See docs/column-id-versioning-design.md §4.1.
CREATE TABLE IF NOT EXISTS ducklake_view (
    view_id BIGINT,
    view_uuid UUID,
    begin_snapshot BIGINT,
    end_snapshot BIGINT,
    schema_id BIGINT,
    view_name VARCHAR,
    dialect VARCHAR,
    sql VARCHAR,
    column_aliases VARCHAR
);

CREATE TABLE IF NOT EXISTS ducklake_column (
    column_id BIGINT,
    begin_snapshot BIGINT,
    end_snapshot BIGINT,
    table_id BIGINT,
    column_order BIGINT,
    column_name VARCHAR,
    column_type VARCHAR,
    initial_default VARCHAR,
    default_value VARCHAR,
    nulls_allowed BOOLEAN,
    parent_column BIGINT,
    default_value_type VARCHAR,
    default_value_dialect VARCHAR
);

-- `partial_max` (DuckLake v1.0) marks a *partial data file* produced by
-- `merge_adjacent_files`: it is the maximum origin snapshot id among the file's
-- merged rows, whose per-row origin is stored in the embedded
-- `_ducklake_internal_snapshot_id` parquet column. NULL for ordinary files
-- (single-origin appends, and `rewrite_data_files` outputs).
CREATE TABLE IF NOT EXISTS ducklake_data_file (
    data_file_id INTEGER PRIMARY KEY,
    table_id INTEGER NOT NULL,
    path VARCHAR NOT NULL,
    path_is_relative BOOLEAN NOT NULL DEFAULT 1,
    file_size_bytes INTEGER NOT NULL,
    footer_size INTEGER,
    encryption_key VARCHAR,
    record_count INTEGER,
    row_id_start INTEGER,
    mapping_id INTEGER,
    begin_snapshot INTEGER NOT NULL,
    end_snapshot INTEGER,
    partial_max INTEGER,
    -- References ducklake_partition_info.partition_id: the partition spec this
    -- file was written under. NULL for files of an unpartitioned table.
    partition_id INTEGER
);

-- Per-snapshot change ledger (DuckLake spec). One row per snapshot recording
-- what the commit did, as a comma-separated `changes_made` string.
CREATE TABLE IF NOT EXISTS ducklake_snapshot_changes (
    snapshot_id INTEGER PRIMARY KEY,
    changes_made VARCHAR,
    author VARCHAR,
    commit_message VARCHAR,
    commit_extra_info VARCHAR
);

-- Per-table row-lineage counter (DuckLake spec). `next_row_id` is the
-- monotonic rowid allocator: a new data file gets its `row_id_start` from
-- the current value, then we advance by `record_count` in the same
-- transaction. `record_count` and `file_size_bytes` mirror the currently-
-- visible totals so DuckDB's `ducklake_table_info` aggregate sees correct
-- numbers for tables we wrote.
CREATE TABLE IF NOT EXISTS ducklake_table_stats (
    table_id INTEGER PRIMARY KEY,
    record_count INTEGER NOT NULL DEFAULT 0,
    next_row_id INTEGER NOT NULL DEFAULT 0,
    file_size_bytes INTEGER NOT NULL DEFAULT 0
);

-- Per-file, per-column statistics (DuckLake spec zone maps). Powers file-level
-- pruning: min/max are the DuckDB-canonical VARCHAR encoding of the bounds,
-- value_count is the non-null count, null_count the null count. Column set
-- mirrors the official extension exactly (no PK, matching upstream). A brand-
-- new table for pre-existing catalogs: `CREATE TABLE IF NOT EXISTS` (re-run by
-- initialize_schema on every open) creates it, which is also what lets DuckDB
-- attach a catalog we wrote (its global-stats query joins the two stats tables).
CREATE TABLE IF NOT EXISTS ducklake_file_column_stats (
    data_file_id INTEGER NOT NULL,
    table_id INTEGER NOT NULL,
    column_id INTEGER NOT NULL,
    column_size_bytes INTEGER,
    value_count INTEGER,
    null_count INTEGER,
    min_value VARCHAR,
    max_value VARCHAR,
    contains_nan BOOLEAN,
    extra_stats VARCHAR
);

-- Table-wide per-column roll-up (DuckLake spec). Feeds the query optimizer
-- (cardinality/column bounds), not file pruning. `contains_null` is a boolean
-- (any null across the table), not a count; min/max widen across all files.
CREATE TABLE IF NOT EXISTS ducklake_table_column_stats (
    table_id INTEGER NOT NULL,
    column_id INTEGER NOT NULL,
    contains_null BOOLEAN,
    contains_nan BOOLEAN,
    min_value VARCHAR,
    max_value VARCHAR,
    extra_stats VARCHAR
);

CREATE TABLE IF NOT EXISTS ducklake_delete_file (
    delete_file_id INTEGER PRIMARY KEY,
    data_file_id INTEGER NOT NULL,
    table_id INTEGER NOT NULL,
    path VARCHAR NOT NULL,
    path_is_relative BOOLEAN NOT NULL DEFAULT 1,
    file_size_bytes INTEGER NOT NULL,
    footer_size INTEGER,
    encryption_key VARCHAR,
    delete_count INTEGER,
    begin_snapshot INTEGER NOT NULL,
    end_snapshot INTEGER,
    -- Max embedded per-row snapshot of a cumulative delete file (current
    -- DuckLake spec). This crate's writer emits per-snapshot delete files and
    -- leaves it NULL; readers use it to window cumulative files by row.
    partial_max INTEGER
);

-- Files queued for physical deletion by the two-phase vacuum (DuckLake spec).
-- `expire_snapshots` GCs unreachable catalog rows and records the orphaned
-- physical paths here; `cleanup_old_files` deletes the objects and removes
-- these rows. `path` is stored relative to the catalog `data_path` root
-- (i.e. already resolved through schema/table) so cleanup needs only a
-- single-level join with `data_path`. Mirrors the upstream
-- `ducklake_files_scheduled_for_deletion` table.
CREATE TABLE IF NOT EXISTS ducklake_files_scheduled_for_deletion (
    data_file_id INTEGER NOT NULL,
    path VARCHAR NOT NULL,
    path_is_relative BOOLEAN NOT NULL DEFAULT 1,
    schedule_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Every file id allocation reads the largest id scheduled here; indexed, that is
-- one index search rather than a scan of every scheduled file.
CREATE INDEX IF NOT EXISTS idx_files_scheduled_for_deletion_file_id
    ON ducklake_files_scheduled_for_deletion (data_file_id);

-- Partition spec generations (DuckLake spec). One row per version of a table's
-- partition spec; `end_snapshot` NULL means currently active. Setting or
-- resetting the spec ends the live row and (for SET) starts a new one, exactly
-- like column versioning. `partition_id` is a catalog-wide id (this crate
-- allocates it from the `next_partition_id` metadata counter).
CREATE TABLE IF NOT EXISTS ducklake_partition_info (
    partition_id INTEGER NOT NULL,
    table_id INTEGER NOT NULL,
    begin_snapshot INTEGER NOT NULL,
    end_snapshot INTEGER
);

-- Partition-key columns for a partition spec (DuckLake spec). One row per key,
-- ordered by `partition_key_index` (0-based). `transform` is the transform
-- string (`identity`, `year`, `month`, `day`, `hour`, `bucket(N)`).
CREATE TABLE IF NOT EXISTS ducklake_partition_column (
    partition_id INTEGER NOT NULL,
    table_id INTEGER NOT NULL,
    partition_key_index INTEGER NOT NULL,
    column_id INTEGER NOT NULL,
    transform VARCHAR NOT NULL
);

-- Per-file partition values (DuckLake spec). The single transformed value that
-- every row in `data_file_id` shares for partition key `partition_key_index`,
-- encoded as a DuckDB-canonical VARCHAR (NULL is a legal partition value).
CREATE TABLE IF NOT EXISTS ducklake_file_partition_value (
    data_file_id INTEGER NOT NULL,
    table_id INTEGER NOT NULL,
    partition_key_index INTEGER NOT NULL,
    partition_value VARCHAR
);

-- The partition pre-filter (`crate::stats_filter`) reads this table once per
-- listing, restricted to one table's rows for one partition key; unindexed that
-- is a scan of every table's partition values. Both columns are fixed-width
-- deliberately: `partition_value` is unbounded text.
CREATE INDEX IF NOT EXISTS idx_file_partition_value_table_key
    ON ducklake_file_partition_value (table_id, partition_key_index);

-- Sort spec (DuckLake spec). Snapshot-versioned like the partition spec:
-- `end_snapshot` NULL means currently active; SET/RESET ends the live row and
-- (for SET) starts a new one. Unlike partitioning, a sort-spec change does NOT
-- bump `schema_version`. `sort_id` is allocated from the `next_sort_id` counter.
CREATE TABLE IF NOT EXISTS ducklake_sort_info (
    sort_id INTEGER NOT NULL,
    table_id INTEGER NOT NULL,
    begin_snapshot INTEGER NOT NULL,
    end_snapshot INTEGER
);

-- Sort-key expressions for a sort spec (DuckLake spec). One row per key, ordered
-- by `sort_key_index` (0-based). `expression` is a sort expression in `dialect`
-- (this crate produces bare column names under the `duckdb` dialect);
-- `sort_direction` is `ASC`/`DESC`; `null_order` is `NULLS_FIRST`/`NULLS_LAST`.
CREATE TABLE IF NOT EXISTS ducklake_sort_expression (
    sort_id INTEGER NOT NULL,
    table_id INTEGER NOT NULL,
    sort_key_index INTEGER NOT NULL,
    expression VARCHAR NOT NULL,
    dialect VARCHAR NOT NULL,
    sort_direction VARCHAR NOT NULL,
    null_order VARCHAR NOT NULL
);
"#;

/// Created beside the inlined-data tables so a catalog that predates tags gains them when
/// opened with `new`, not only with `new_with_init`.
const SQL_CREATE_TAG_TABLES: &str = r#"
CREATE TABLE IF NOT EXISTS ducklake_tag (
    object_id BIGINT,
    begin_snapshot BIGINT,
    end_snapshot BIGINT,
    key VARCHAR,
    value VARCHAR
);

CREATE TABLE IF NOT EXISTS ducklake_column_tag (
    table_id BIGINT,
    column_id BIGINT,
    begin_snapshot BIGINT,
    end_snapshot BIGINT,
    key VARCHAR,
    value VARCHAR
);
"#;

const SQL_CREATE_INLINED_DATA_TABLES: &str =
    "CREATE TABLE IF NOT EXISTS ducklake_inlined_data_tables (
    table_id INTEGER,
    table_name VARCHAR,
    schema_version INTEGER
);";

/// SQLite-based metadata writer for DuckLake catalogs.
#[derive(Debug, Clone)]
pub struct SqliteMetadataWriter {
    pool: SqlitePool,
    lock_path: Option<PathBuf>,
    commit_lock: Arc<Mutex<()>>,
}

impl SqliteMetadataWriter {
    pub async fn new(connection_string: &str) -> Result<Self> {
        Self::with_max_connections(connection_string, DEFAULT_MAX_CONNECTIONS).await
    }

    pub async fn with_max_connections(
        connection_string: &str,
        max_connections: u32,
    ) -> Result<Self> {
        let options = SqliteConnectOptions::from_str(connection_string)?;
        let filename = options.get_filename();
        let lock_path = (!filename.as_os_str().is_empty()
            && filename != std::path::Path::new(":memory:"))
        .then(|| {
            let mut path = filename.as_os_str().to_os_string();
            path.push(".commit.lock");
            PathBuf::from(path)
        });
        let url = connection_string.to_string();
        let pool = crate::metadata_provider::connect_on_catalog_runtime(async move {
            SqlitePoolOptions::new()
                .max_connections(max_connections)
                .connect(&url)
                .await
        })
        .await?;

        // Existing catalogs must migrate even when callers skip `initialize_schema`
        migrate_snapshot_changes_nullable(&pool).await?;
        sqlx::query(SQL_CREATE_INLINED_DATA_TABLES)
            .execute(&pool)
            .await?;
        sqlx::query(SQL_CREATE_TAG_TABLES).execute(&pool).await?;
        Ok(Self {
            pool,
            lock_path,
            commit_lock: Arc::new(Mutex::new(())),
        })
    }

    pub async fn new_with_init(connection_string: &str) -> Result<Self> {
        let writer = Self::new(connection_string).await?;
        writer.initialize_schema()?;
        Ok(writer)
    }

    /// Tombstone a live table at a new "drop snapshot".
    ///
    /// Allocates a fresh snapshot and sets `end_snapshot = <drop snapshot>` on the
    /// currently-live rows for the named table in `ducklake_table`,
    /// `ducklake_column`, `ducklake_data_file`, and `ducklake_delete_file`. Reads at
    /// any snapshot `>=` the drop snapshot no longer see the table; earlier snapshots
    /// are unaffected (time travel preserved). This is the single-catalog mirror of
    /// [`crate::multicatalog::MulticatalogManager::drop_table_in_catalog`].
    ///
    /// Idempotent: returns `Ok(false)` (no snapshot allocated) when no live
    /// `(schema, table)` pair exists. Physical data files are not removed — that is
    /// the job of [`Self::expire_snapshots`] + [`crate::maintenance::cleanup_old_files_sqlite`].
    ///
    /// The table's `ducklake_table_stats` row is intentionally left in place:
    /// `next_row_id` must stay monotonic across the table's lifetime, and a
    /// recreate-after-drop gets a fresh `table_id`. The orphaned stats row is
    /// reclaimed by [`Self::expire_snapshots`] once the table is fully expired.
    pub fn drop_table(&self, schema_name: &str, table_name: &str) -> Result<bool> {
        validate_name(schema_name, "Schema")?;
        validate_name(table_name, "Table")?;
        block_on(async {
            let mut tx = self.pool.begin().await?;

            // Take the write lock up front (write-lock-first invariant; see
            // `insert_snapshot`) by allocating the snapshot before any read, so a
            // concurrent drop/promote blocks on the lock rather than failing a
            // deferred read→write upgrade. A drop is DDL → also bump schema_version.
            // No ducklake_schema_versions row: the table has no live schema
            // afterward (matches upstream `DropTables` and
            // `multicatalog.rs::drop_table_in_catalog`).
            let (drop_snapshot, _schema_version) = insert_snapshot(&mut tx).await?;

            // Resolve the live table_id. `end_snapshot IS NULL` on both schema and
            // table makes an already-dropped table a no-op (idempotent): we return
            // without committing, so `tx` rolls back and the snapshot allocated
            // above leaves no trace.
            let table_id: i64 = match sqlx::query(
                "SELECT t.table_id FROM ducklake_table t
                 JOIN ducklake_schema s ON s.schema_id = t.schema_id
                 WHERE s.schema_name = ? AND s.end_snapshot IS NULL
                   AND t.table_name = ? AND t.end_snapshot IS NULL",
            )
            .bind(schema_name)
            .bind(table_name)
            .fetch_optional(&mut *tx)
            .await?
            {
                Some(r) => r.try_get(0)?,
                None => return Ok(false),
            };

            bump_schema_version(&mut tx, drop_snapshot).await?;

            for child in [
                "ducklake_table",
                "ducklake_column",
                "ducklake_column_tag",
                "ducklake_data_file",
                "ducklake_delete_file",
            ] {
                sqlx::query(AssertSqlSafe(format!(
                    "UPDATE {child} SET end_snapshot = ?
                     WHERE table_id = ? AND end_snapshot IS NULL"
                )))
                .bind(drop_snapshot)
                .bind(table_id)
                .execute(&mut *tx)
                .await?;
            }

            sqlx::query(
                "UPDATE ducklake_tag SET end_snapshot = ?
                 WHERE object_id = ? AND end_snapshot IS NULL",
            )
            .bind(drop_snapshot)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            record_snapshot_changes(
                &mut tx,
                drop_snapshot,
                &format!("dropped_table:{table_id}"),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            tx.commit().await?;
            Ok(true)
        })
    }

    /// Expire snapshots and garbage-collect the catalog metadata they leave behind.
    ///
    /// Ports the official `ducklake_expire_snapshots` / `DuckLakeMetadataManager::DeleteSnapshots`.
    /// The most recent snapshot is never expired. After deleting the chosen snapshot
    /// rows, every table / data file / delete file no longer reachable by any surviving
    /// snapshot is removed from the catalog and its physical path is recorded in
    /// `ducklake_files_scheduled_for_deletion` for later physical deletion via
    /// [`crate::maintenance::cleanup_old_files_sqlite`]. A fully expired table's
    /// inlined data and inlined delete tables are dropped with its registry rows
    /// (see `crate::inlined_reclaim`); inlined rows no surviving snapshot can read
    /// stay in tables still live, as official keeps them. Returns the expired
    /// snapshots.
    ///
    /// Reachability is global here — correct for a single-catalog SQLite metadata DB.
    /// The whole operation runs in one transaction, so a crash can't leave scheduled
    /// rows out of sync with the catalog.
    pub fn expire_snapshots(&self, criteria: ExpireCriteria) -> Result<Vec<ExpiredSnapshot>> {
        block_on(async {
            let mut tx = self.pool.begin().await?;

            // The most recent snapshot is never expirable.
            let most_recent: Option<i64> =
                sqlx::query("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
                    .fetch_one(&mut *tx)
                    .await?
                    .try_get(0)?;
            let most_recent = match most_recent {
                Some(id) => id,
                None => {
                    tx.commit().await?;
                    return Ok(Vec::new());
                },
            };

            // 1. Resolve the snapshots to expire (excluding the most recent).
            let candidates: Vec<ExpiredSnapshot> = match &criteria {
                ExpireCriteria::Versions(versions) => {
                    let ids: Vec<i64> = versions
                        .iter()
                        .copied()
                        .filter(|&v| v != most_recent)
                        .collect();
                    if ids.is_empty() {
                        tx.commit().await?;
                        return Ok(Vec::new());
                    }
                    let rows = sqlx::query(AssertSqlSafe(format!(
                        "SELECT snapshot_id, snapshot_time FROM ducklake_snapshot
                         WHERE snapshot_id IN ({}) ORDER BY snapshot_id",
                        id_list(&ids)
                    )))
                    .fetch_all(&mut *tx)
                    .await?;
                    rows_to_snapshots(rows)?
                },
                ExpireCriteria::OlderThan(ts) => {
                    let rows = sqlx::query(
                        "SELECT snapshot_id, snapshot_time FROM ducklake_snapshot
                         WHERE snapshot_id != ? AND snapshot_time < ? ORDER BY snapshot_id",
                    )
                    .bind(most_recent)
                    .bind(format_sql_timestamp(ts))
                    .fetch_all(&mut *tx)
                    .await?;
                    rows_to_snapshots(rows)?
                },
            };
            if candidates.is_empty() {
                tx.commit().await?;
                return Ok(Vec::new());
            }
            let expire_ids: Vec<i64> = candidates.iter().map(|s| s.snapshot_id).collect();

            // 2. Delete the snapshot rows themselves, and the per-commit change
            //    records that hang off them. Official deletes both in one loop
            //    (DeleteSnapshots); a change record outlives its snapshot otherwise.
            for table in ["ducklake_snapshot", "ducklake_snapshot_changes"] {
                sqlx::query(AssertSqlSafe(format!(
                    "DELETE FROM {table} WHERE snapshot_id IN ({})",
                    id_list(&expire_ids)
                )))
                .execute(&mut *tx)
                .await?;
            }

            // 3. Tables whose lifetime is no longer covered by any surviving snapshot
            //    AND which have no surviving live version.
            let dead_tables: Vec<i64> = sqlx::query(
                "SELECT t.table_id FROM ducklake_table t
                 WHERE t.end_snapshot IS NOT NULL AND NOT EXISTS (
                     SELECT 1 FROM ducklake_snapshot
                     WHERE snapshot_id >= t.begin_snapshot AND snapshot_id < t.end_snapshot)
                 AND NOT EXISTS (
                     SELECT 1 FROM ducklake_table t2
                     WHERE t2.table_id = t.table_id
                       AND (t2.end_snapshot IS NULL OR EXISTS (
                           SELECT 1 FROM ducklake_snapshot
                           WHERE snapshot_id >= t2.begin_snapshot
                             AND snapshot_id < t2.end_snapshot)))",
            )
            .fetch_all(&mut *tx)
            .await?
            .into_iter()
            .map(|r| r.try_get::<i64, _>(0))
            .collect::<std::result::Result<_, _>>()?;
            let dead_table_filter = if dead_tables.is_empty() {
                "0".to_string()
            } else {
                format!("df.table_id IN ({})", id_list(&dead_tables))
            };

            // 4. Data files no longer referenced by any surviving snapshot (or belonging
            //    to a dead table): schedule their physical paths, then drop the rows.
            let dead_data_files = sqlx::query(AssertSqlSafe(format!(
                "SELECT df.data_file_id, {RESOLVED_PATH} AS resolved_path, {REL_FLAG} AS rel
                 FROM ducklake_data_file df
                 JOIN ducklake_table t ON t.table_id = df.table_id
                 JOIN ducklake_schema s ON s.schema_id = t.schema_id
                 WHERE ({dead_table_filter}) OR (df.end_snapshot IS NOT NULL AND NOT EXISTS (
                     SELECT 1 FROM ducklake_snapshot
                     WHERE snapshot_id >= df.begin_snapshot AND snapshot_id < df.end_snapshot))"
            )))
            .fetch_all(&mut *tx)
            .await?;
            let data_file_ids = schedule_files(&mut tx, dead_data_files).await?;
            if !data_file_ids.is_empty() {
                // Per-file metadata dies with its file, keyed on the same ids and in
                // the same group official uses (DeleteSnapshots). The compaction path
                // below already does this for the sources it retires; without it here
                // the rows are unreachable but never reclaimed.
                // ducklake_file_variant_stats is the fourth table official deletes
                // here. Not in this schema, so omitted. IF IT IS EVER ADDED, ADD IT
                // HERE TOO — a per-file table expire does not reclaim orphans
                // silently and forever.
                for table in [
                    "ducklake_data_file",
                    "ducklake_file_column_stats",
                    "ducklake_file_partition_value",
                ] {
                    sqlx::query(AssertSqlSafe(format!(
                        "DELETE FROM {table} WHERE data_file_id IN ({})",
                        id_list(&data_file_ids)
                    )))
                    .execute(&mut *tx)
                    .await?;
                }
            }

            // 5. Delete files orphaned by the data files above, by a dead table, or no
            //    longer referenced by any surviving snapshot. (Our writer does not emit
            //    delete files yet, so this is a no-op for catalogs we produce.)
            let dead_data_filter = if data_file_ids.is_empty() {
                "0".to_string()
            } else {
                format!("df.data_file_id IN ({})", id_list(&data_file_ids))
            };
            let dead_delete_table_filter = if dead_tables.is_empty() {
                "0".to_string()
            } else {
                format!("df.table_id IN ({})", id_list(&dead_tables))
            };
            let dead_delete_files = sqlx::query(AssertSqlSafe(format!(
                "SELECT df.delete_file_id, {RESOLVED_PATH} AS resolved_path, {REL_FLAG} AS rel
                 FROM ducklake_delete_file df
                 JOIN ducklake_table t ON t.table_id = df.table_id
                 JOIN ducklake_schema s ON s.schema_id = t.schema_id
                 WHERE ({dead_data_filter}) OR ({dead_delete_table_filter})
                    OR (df.end_snapshot IS NOT NULL AND NOT EXISTS (
                        SELECT 1 FROM ducklake_snapshot
                        WHERE snapshot_id >= df.begin_snapshot AND snapshot_id < df.end_snapshot))"
            )))
            .fetch_all(&mut *tx)
            .await?;
            let delete_file_ids = schedule_files(&mut tx, dead_delete_files).await?;
            if !delete_file_ids.is_empty() {
                sqlx::query(AssertSqlSafe(format!(
                    "DELETE FROM ducklake_delete_file WHERE delete_file_id IN ({})",
                    id_list(&delete_file_ids)
                )))
                .execute(&mut *tx)
                .await?;
            }

            // 6. Reclaim per-table metadata for fully-expired tables (this is where a
            //    dropped table's orphaned ducklake_table_stats row is removed).
            if !dead_tables.is_empty() {
                let dead = id_list(&dead_tables);
                // The table's inlined storage goes with it: its registered inlined data
                // tables and its inlined delete table, listed before the registry rows are
                // deleted below and dropped after them, as official does.
                let registered = sqlx::query_scalar::<_, String>(AssertSqlSafe(format!(
                    "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id IN ({dead})"
                )))
                .fetch_all(&mut *tx)
                .await?;
                let inlined_storage = dead_table_inlined_storage(registered, &dead_tables)?;
                // `ducklake_schema_versions` is keyed by table_id and has no
                // end_snapshot; reclaim its rows here once the table is fully
                // expired (a recreate-after-drop gets a fresh table_id, so this
                // can't strand a live table's ledger). Matches the Postgres
                // multicatalog cleanup and upstream GC.
                for table in [
                    "ducklake_table",
                    "ducklake_table_stats",
                    "ducklake_table_column_stats",
                    "ducklake_partition_info",
                    "ducklake_partition_column",
                    "ducklake_column",
                    "ducklake_sort_info",
                    "ducklake_sort_expression",
                    "ducklake_schema_versions",
                    "ducklake_inlined_data_tables",
                    "ducklake_column_tag",
                ] {
                    sqlx::query(AssertSqlSafe(format!(
                        "DELETE FROM {table} WHERE table_id IN ({dead})"
                    )))
                    .execute(&mut *tx)
                    .await?;
                }
                for table in inlined_storage {
                    sqlx::query(AssertSqlSafe(format!(
                        "DROP TABLE IF EXISTS {}",
                        quote_ident(&table)
                    )))
                    .execute(&mut *tx)
                    .await?;
                }
            }

            // Object tags ended before every surviving snapshot can no longer be read.
            // Column tags follow their table instead, as in official: they go with a dead
            // table above, and an ended tag of a live table is kept.
            sqlx::query(
                "DELETE FROM ducklake_tag
                 WHERE end_snapshot IS NOT NULL AND NOT EXISTS (
                     SELECT 1 FROM ducklake_snapshot
                     WHERE snapshot_id >= ducklake_tag.begin_snapshot
                       AND snapshot_id < ducklake_tag.end_snapshot)",
            )
            .execute(&mut *tx)
            .await?;

            // 7. Reclaim schemas no longer covered by any surviving snapshot.
            sqlx::query(
                "DELETE FROM ducklake_schema
                 WHERE end_snapshot IS NOT NULL AND NOT EXISTS (
                     SELECT 1 FROM ducklake_snapshot
                     WHERE snapshot_id >= ducklake_schema.begin_snapshot
                       AND snapshot_id < ducklake_schema.end_snapshot)",
            )
            .execute(&mut *tx)
            .await?;

            tx.commit().await?;
            Ok(candidates)
        })
    }

    /// List files scheduled for physical deletion, optionally filtered by schedule time.
    pub(crate) fn list_scheduled_for_deletion(
        &self,
        criteria: &CleanupCriteria,
    ) -> Result<Vec<ScheduledFile>> {
        block_on(async {
            let rows = match criteria {
                CleanupCriteria::All => {
                    sqlx::query(
                        "SELECT data_file_id, path, path_is_relative
                     FROM ducklake_files_scheduled_for_deletion",
                    )
                    .fetch_all(&self.pool)
                    .await?
                },
                CleanupCriteria::OlderThan(ts) => {
                    sqlx::query(
                        "SELECT data_file_id, path, path_is_relative
                     FROM ducklake_files_scheduled_for_deletion
                     WHERE schedule_start < ?",
                    )
                    .bind(format_sql_timestamp(ts))
                    .fetch_all(&self.pool)
                    .await?
                },
            };
            rows.into_iter()
                .map(|r| {
                    Ok(ScheduledFile {
                        data_file_id: r.try_get(0)?,
                        path: r.try_get(1)?,
                        path_is_relative: r.try_get::<i64, _>(2)? != 0,
                        // Single-catalog layout: there is no other catalog for a
                        // file to belong to, so every scheduled row is this
                        // catalog's own.
                        scheduled_by_owner: None,
                    })
                })
                .collect()
        })
    }

    /// Remove scheduled-deletion bookkeeping rows after their objects are gone.
    pub(crate) fn remove_scheduled(&self, ids: &[i64]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        block_on(async {
            sqlx::query(AssertSqlSafe(format!(
                "DELETE FROM ducklake_files_scheduled_for_deletion WHERE data_file_id IN ({})",
                id_list(ids)
            )))
            .execute(&self.pool)
            .await?;
            Ok(())
        })
    }

    /// Every physical file currently referenced by the catalog: live + tombstoned
    /// data files, live + tombstoned delete files, and the still-scheduled
    /// deletion queue (those rows haven't been processed by `cleanup_old_files`
    /// yet, so the files might still exist on disk and must not be touched by
    /// orphan cleanup). Each row is `(path, path_is_relative)` resolved relative
    /// to the catalog `data_path` root; the caller resolves against the actual
    /// `data_path` for comparison with object_store keys.
    ///
    /// Used by [`crate::maintenance::delete_orphaned_files_sqlite`].
    pub(crate) fn list_referenced_paths(&self) -> Result<Vec<(String, bool)>> {
        block_on(async {
            let q = format!(
                "SELECT {RESOLVED_PATH} AS p, {REL_FLAG} AS rel
                 FROM ducklake_data_file df
                 JOIN ducklake_table t ON t.table_id = df.table_id
                 JOIN ducklake_schema s ON s.schema_id = t.schema_id
                 UNION ALL
                 SELECT {RESOLVED_PATH} AS p, {REL_FLAG} AS rel
                 FROM ducklake_delete_file df
                 JOIN ducklake_table t ON t.table_id = df.table_id
                 JOIN ducklake_schema s ON s.schema_id = t.schema_id
                 UNION ALL
                 SELECT path AS p, CAST(path_is_relative AS INTEGER) AS rel
                 FROM ducklake_files_scheduled_for_deletion"
            );
            let rows = sqlx::query(AssertSqlSafe(q.as_str()))
                .fetch_all(&self.pool)
                .await?;
            rows.into_iter()
                .map(|r| {
                    let p: String = r.try_get(0)?;
                    let rel: i64 = r.try_get(1)?;
                    Ok((p, rel != 0))
                })
                .collect()
        })
    }
}

/// SQL expression yielding a file path resolved relative to the catalog `data_path`
/// root, mirroring the read-side hierarchical resolution (file → table → schema →
/// data_path). An absolute path anywhere in the chain short-circuits to that absolute
/// path. Assumes `/`-joined relative names (everything our writer produces).
const RESOLVED_PATH: &str = "CASE
    WHEN NOT df.path_is_relative THEN df.path
    WHEN NOT t.path_is_relative THEN RTRIM(t.path, '/') || '/' || LTRIM(df.path, '/')
    ELSE RTRIM(s.path, '/') || '/' || RTRIM(t.path, '/') || '/' || LTRIM(df.path, '/')
END";

/// Companion to [`RESOLVED_PATH`]: 1 only when the whole chain is relative (so the
/// resolved path is relative to `data_path`), else 0 (the resolved path is absolute).
const REL_FLAG: &str =
    "(CASE WHEN df.path_is_relative AND t.path_is_relative AND s.path_is_relative
           THEN 1 ELSE 0 END)";

/// Map `(snapshot_id, snapshot_time)` rows to [`ExpiredSnapshot`]s.
fn rows_to_snapshots(rows: Vec<sqlx::sqlite::SqliteRow>) -> Result<Vec<ExpiredSnapshot>> {
    rows.into_iter()
        .map(|r| {
            Ok(ExpiredSnapshot {
                snapshot_id: r.try_get(0)?,
                snapshot_time: r.try_get(1)?,
            })
        })
        .collect()
}

/// Insert `(id, resolved_path, rel)` rows (as produced by [`RESOLVED_PATH`]/[`REL_FLAG`])
/// into `ducklake_files_scheduled_for_deletion` and return the ids scheduled.
async fn schedule_files(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    rows: Vec<sqlx::sqlite::SqliteRow>,
) -> Result<Vec<i64>> {
    let mut ids = Vec::with_capacity(rows.len());
    for row in rows {
        let id: i64 = row.try_get(0)?;
        let path: String = row.try_get(1)?;
        let rel: i64 = row.try_get(2)?;
        sqlx::query(
            "INSERT INTO ducklake_files_scheduled_for_deletion
                 (data_file_id, path, path_is_relative, schedule_start)
             VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
        )
        .bind(id)
        .bind(&path)
        .bind(rel)
        .execute(&mut **tx)
        .await?;
        ids.push(id);
    }
    Ok(ids)
}

/// Atomically reserve `n` consecutive ids from a monotonic counter stored in
/// `ducklake_metadata` (seeded by `initialize_schema`), returning the LAST id of
/// the block — the reserved ids are `last - n + 1 ..= last`. The `UPDATE`
/// serializes under SQLite's single-writer lock, so concurrent writers (even to
/// different tables) never hand out overlapping ids. Used for `snapshot_id` and
/// `column_id`, which are reserved in `begin_write_transaction` and inserted at
/// the commit: rowid autoincrement can't be used there because inserting the
/// `ducklake_snapshot` row would advance the head (`MAX(snapshot_id)`) before the
/// data is committed.
/// The next data or delete file id: one counter (`next_file_id` in
/// `ducklake_metadata`) for both, as official's `next_file_id`, so an id is
/// never handed out twice. SQLite's row ids would reuse the largest id once
/// expiry removed its row, and a deletion still recorded against the old file
/// (an inlined deletion, which official keeps after the file goes) would then
/// apply to the new one. A catalog without the counter seeds it past every id
/// in use, its inlined deletions' included. Each id is also past every data,
/// delete and scheduled file id in use at the time, so a file row a writer
/// without the counter inserted (an older release) is never collided with.
async fn next_file_id(conn: &mut sqlx::SqliteConnection) -> Result<i64> {
    let seeded: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM ducklake_metadata WHERE key = 'next_file_id' AND scope IS NULL)",
    )
    .fetch_one(&mut *conn)
    .await?;
    if !seeded {
        let mut highest: i64 = sqlx::query_scalar(
            "SELECT MAX(COALESCE((SELECT MAX(data_file_id) FROM ducklake_data_file), 0),
                        COALESCE((SELECT MAX(delete_file_id) FROM ducklake_delete_file), 0),
                        COALESCE((SELECT MAX(data_file_id) FROM ducklake_files_scheduled_for_deletion), 0))",
        )
        .fetch_one(&mut *conn)
        .await?;
        let deletion_tables: Vec<String> = sqlx::query_scalar(
            "SELECT name FROM sqlite_master
             WHERE type = 'table' AND name LIKE 'ducklake\\_inlined\\_delete\\_%' ESCAPE '\\'",
        )
        .fetch_all(&mut *conn)
        .await?;
        for table in deletion_tables {
            // A table of another layout, without `file_id`, holds no id.
            let max: Option<i64> = sqlx::query_scalar(AssertSqlSafe(format!(
                "SELECT MAX(file_id) FROM {}",
                quote_ident(&table)
            )))
            .fetch_one(&mut *conn)
            .await
            .unwrap_or(None);
            highest = highest.max(max.unwrap_or(0));
        }
        sqlx::query(
            "INSERT INTO ducklake_metadata (key, value, scope) VALUES ('next_file_id', ?, NULL)",
        )
        .bind(highest.to_string())
        .execute(&mut *conn)
        .await?;
    }
    let id: i64 = sqlx::query_scalar(
        "UPDATE ducklake_metadata
         SET value = CAST(MAX(CAST(value AS INTEGER),
                              COALESCE((SELECT MAX(data_file_id) FROM ducklake_data_file), 0),
                              COALESCE((SELECT MAX(delete_file_id) FROM ducklake_delete_file), 0),
                              COALESCE((SELECT MAX(data_file_id)
                                        FROM ducklake_files_scheduled_for_deletion), 0)) + 1
                         AS TEXT)
         WHERE key = 'next_file_id' AND scope IS NULL
         RETURNING CAST(value AS INTEGER)",
    )
    .fetch_one(&mut *conn)
    .await?;
    Ok(id)
}

async fn reserve_ids(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    key: &str,
    n: i64,
) -> Result<i64> {
    let last: i64 = sqlx::query(
        "UPDATE ducklake_metadata
         SET value = CAST(CAST(value AS INTEGER) + ? AS TEXT)
         WHERE key = ? AND scope IS NULL
         RETURNING CAST(value AS INTEGER)",
    )
    .bind(n)
    .bind(key)
    .fetch_one(&mut **tx)
    .await?
    .try_get(0)?;
    Ok(last)
}

/// Upgrade an existing `ducklake_column` from the legacy shape
/// (`column_id INTEGER PRIMARY KEY` — this library's pre-versioning layout) to
/// upstream's bare shape (`column_id BIGINT`, no PK; see `SQL_CREATE_SCHEMA`).
///
/// `CREATE TABLE IF NOT EXISTS` only shapes *new* catalogs; a catalog written by
/// an earlier version keeps its old table, so we must rebuild it in place — SQLite
/// cannot `ALTER TABLE ... DROP PRIMARY KEY`. The single-row PK forbade a second
/// row sharing a `column_id`, which a versioned / type-promoted column requires,
/// so this migration is what lets existing users' catalogs support type promotion.
///
/// **Idempotent:** a no-op once `column_id` is no longer a PK and all current
/// metadata columns exist (re-runs on every `initialize_schema`). **Crash-safe:**
/// SQLite runs the rebuild in one transaction, so a crash leaves either the old or
/// the new table whole, never a half-state. **Lossless:** every row and every
/// `column_id` value is preserved (field-ids baked in Parquet stay valid), and it
/// tolerates older catalogs missing some of the 13 columns (those are filled with
/// `NULL`).
async fn migrate_ducklake_column_drop_pk(pool: &SqlitePool) -> Result<()> {
    // Legacy shape iff `column_id` is a PRIMARY KEY. `pragma_table_info` returns
    // one row per column with a `pk` flag (0 = not part of the PK).
    let info = sqlx::query("SELECT name, pk FROM pragma_table_info('ducklake_column')")
        .fetch_all(pool)
        .await?;
    let mut legacy_cols: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut column_id_is_pk = false;
    for row in &info {
        let name: String = row.try_get("name")?;
        let pk: i64 = row.try_get("pk")?;
        if name == "column_id" && pk != 0 {
            column_id_is_pk = true;
        }
        legacy_cols.insert(name);
    }
    let has_default_columns =
        ["initial_default", "default_value", "default_value_type", "default_value_dialect"]
            .iter()
            .all(|name| legacy_cols.contains(*name));
    if !column_id_is_pk && has_default_columns {
        return Ok(());
    }

    // Target = upstream's exact bare column set/order. MUST stay in sync with the
    // `ducklake_column` DDL in `SQL_CREATE_SCHEMA`.
    const TARGET: &[&str] = &[
        "column_id",
        "begin_snapshot",
        "end_snapshot",
        "table_id",
        "column_order",
        "column_name",
        "column_type",
        "initial_default",
        "default_value",
        "nulls_allowed",
        "parent_column",
        "default_value_type",
        "default_value_dialect",
    ];
    // Copy each target column from the legacy table if present, else NULL — so a
    // catalog that predates the `*default*`/`parent_column` columns still migrates.
    let select_list = TARGET
        .iter()
        .map(|c| {
            if legacy_cols.contains(*c) {
                (*c).to_string()
            } else {
                "NULL".to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    let insert_list = TARGET.join(", ");

    let mut tx = pool.begin().await?;
    sqlx::query(
        "CREATE TABLE ducklake_column__migrate (
            column_id BIGINT,
            begin_snapshot BIGINT,
            end_snapshot BIGINT,
            table_id BIGINT,
            column_order BIGINT,
            column_name VARCHAR,
            column_type VARCHAR,
            initial_default VARCHAR,
            default_value VARCHAR,
            nulls_allowed BOOLEAN,
            parent_column BIGINT,
            default_value_type VARCHAR,
            default_value_dialect VARCHAR
        )",
    )
    .execute(&mut *tx)
    .await?;
    sqlx::query(AssertSqlSafe(format!(
        "INSERT INTO ducklake_column__migrate ({insert_list}) \
         SELECT {select_list} FROM ducklake_column"
    )))
    .execute(&mut *tx)
    .await?;
    sqlx::query("DROP TABLE ducklake_column")
        .execute(&mut *tx)
        .await?;
    sqlx::query("ALTER TABLE ducklake_column__migrate RENAME TO ducklake_column")
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;
    Ok(())
}

/// Upgrade a pre-existing catalog to track `schema_version`: add the
/// `ducklake_snapshot.schema_version` column (the `ducklake_schema_versions` table
/// is handled by `CREATE TABLE IF NOT EXISTS` in `SQL_CREATE_SCHEMA`). `CREATE
/// TABLE IF NOT EXISTS` only shapes *new* catalogs, and SQLite has no
/// `ADD COLUMN IF NOT EXISTS`, so we probe `pragma_table_info` and add the column
/// only when missing.
///
/// **Idempotent:** a no-op once the column exists (re-runs on every
/// `initialize_schema`). **Lossless:** existing snapshot rows take the column
/// `DEFAULT 0`. We deliberately do NOT fabricate a dense historical
/// `schema_version` / backfill `ducklake_schema_versions`: this library never
/// recorded which past snapshots were DDL, so any backfill would be guessed
/// history. The counter grows correctly from the next DDL commit forward, which is
/// monotonic and honest (old snapshots simply read as version 0).
async fn migrate_add_schema_version(pool: &SqlitePool) -> Result<()> {
    let info = sqlx::query("SELECT name FROM pragma_table_info('ducklake_snapshot')")
        .fetch_all(pool)
        .await?;
    let mut has_schema_version = false;
    for row in &info {
        let name: String = row.try_get("name")?;
        if name == "schema_version" {
            has_schema_version = true;
        }
    }
    if !has_schema_version {
        sqlx::query(
            "ALTER TABLE ducklake_snapshot ADD COLUMN schema_version INTEGER NOT NULL DEFAULT 0",
        )
        .execute(pool)
        .await?;
    }
    Ok(())
}

async fn migrate_add_metadata_scope_columns(pool: &SqlitePool) -> Result<()> {
    let columns = sqlx::query("SELECT name FROM pragma_table_info('ducklake_metadata')")
        .fetch_all(pool)
        .await?;
    let has_scope = columns
        .iter()
        .any(|row| row.get::<String, _>("name") == "scope");
    let has_scope_id = columns
        .iter()
        .any(|row| row.get::<String, _>("name") == "scope_id");
    if !has_scope {
        add_column(
            pool,
            "ducklake_metadata",
            "scope",
            "ALTER TABLE ducklake_metadata ADD COLUMN scope VARCHAR",
        )
        .await?;
    }
    if !has_scope_id {
        add_column(
            pool,
            "ducklake_metadata",
            "scope_id",
            "ALTER TABLE ducklake_metadata ADD COLUMN scope_id INTEGER",
        )
        .await?;
    }
    Ok(())
}

/// Run `alter`, which adds `column` to `table`. SQLite has no `ADD COLUMN IF
/// NOT EXISTS`, so the caller probes first; an initialization that passed the
/// probe while another added the column fails with a duplicate column name,
/// which is accepted when the column now exists.
async fn add_column(pool: &SqlitePool, table: &str, column: &str, alter: &str) -> Result<()> {
    let Err(error) = sqlx::query(AssertSqlSafe(alter.to_string()))
        .execute(pool)
        .await
    else {
        return Ok(());
    };
    let exists: bool =
        sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM pragma_table_info(?) WHERE name = ?)")
            .bind(table)
            .bind(column)
            .fetch_one(pool)
            .await?;
    if !exists {
        return Err(error.into());
    }
    Ok(())
}

/// Add `partial_max` to a pre-existing catalog's `ducklake_data_file` and
/// `ducklake_delete_file` (DuckLake v1.0); catalogs from v0.5.0 and earlier
/// lack the delete-file column. `CREATE TABLE IF NOT EXISTS` only shapes
/// brand-new catalogs, so an older one needs this `ALTER`.
///
/// **Idempotent:** a no-op once the columns exist (re-runs on every
/// `initialize_schema`). **Lossless:** existing rows take the column `NULL`,
/// which is exactly the "not a partial file" value, and a delete file without
/// it holds deletions of its `begin_snapshot` only.
async fn migrate_add_partial_max(pool: &SqlitePool) -> Result<()> {
    for table in ["ducklake_data_file", "ducklake_delete_file"] {
        let info = sqlx::query(AssertSqlSafe(format!(
            "SELECT name FROM pragma_table_info('{table}')"
        )))
        .fetch_all(pool)
        .await?;
        let mut has_partial_max = false;
        for row in &info {
            let name: String = row.try_get("name")?;
            if name == "partial_max" {
                has_partial_max = true;
            }
        }
        if !has_partial_max {
            add_column(
                pool,
                table,
                "partial_max",
                &format!("ALTER TABLE {table} ADD COLUMN partial_max INTEGER"),
            )
            .await?;
        }
    }
    Ok(())
}

/// Upgrade a pre-existing catalog to carry `ducklake_data_file.partition_id`
/// (idempotent, lossless — NULL means "not partitioned", correct for every file
/// written before partitioning support). `CREATE TABLE IF NOT EXISTS` never
/// alters an existing table, so this ALTER is required for old catalogs.
async fn migrate_add_partition_id(pool: &SqlitePool) -> Result<()> {
    let info = sqlx::query("SELECT name FROM pragma_table_info('ducklake_data_file')")
        .fetch_all(pool)
        .await?;
    let mut has_partition_id = false;
    for row in &info {
        let name: String = row.try_get("name")?;
        if name == "partition_id" {
            has_partition_id = true;
        }
    }
    if !has_partition_id {
        sqlx::query("ALTER TABLE ducklake_data_file ADD COLUMN partition_id INTEGER")
            .execute(pool)
            .await?;
    }
    Ok(())
}

/// Give legacy `data_path`, schema path, and table path values the trailing `/`
/// DuckLake readers expect. Idempotent.
async fn migrate_directory_paths(pool: &SqlitePool) -> Result<()> {
    let mut tx = pool.begin().await?;
    sqlx::query(
        "UPDATE ducklake_metadata SET value = value || '/'
         WHERE key = 'data_path' AND scope IS NULL AND value <> '' AND value NOT LIKE '%/'",
    )
    .execute(&mut *tx)
    .await?;
    sqlx::query(
        "UPDATE ducklake_schema SET path = path || '/' WHERE path <> '' AND path NOT LIKE '%/'",
    )
    .execute(&mut *tx)
    .await?;
    sqlx::query(
        "UPDATE ducklake_table SET path = path || '/' WHERE path <> '' AND path NOT LIKE '%/'",
    )
    .execute(&mut *tx)
    .await?;
    tx.commit().await?;
    Ok(())
}

async fn migrate_snapshot_changes_nullable(pool: &SqlitePool) -> Result<()> {
    let info = sqlx::query("SELECT name FROM pragma_table_info('ducklake_snapshot_changes')")
        .fetch_all(pool)
        .await?;
    if info.is_empty() {
        return Ok(());
    }

    let columns = info
        .iter()
        .map(|row| row.try_get::<String, _>("name"))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    for (name, statement) in [
        (
            "author",
            "ALTER TABLE ducklake_snapshot_changes ADD COLUMN author VARCHAR DEFAULT NULL",
        ),
        (
            "commit_message",
            "ALTER TABLE ducklake_snapshot_changes ADD COLUMN commit_message VARCHAR DEFAULT NULL",
        ),
        (
            "commit_extra_info",
            "ALTER TABLE ducklake_snapshot_changes ADD COLUMN commit_extra_info VARCHAR DEFAULT NULL",
        ),
    ] {
        if !columns.iter().any(|column| column == name) {
            sqlx::query(statement).execute(pool).await?;
        }
    }

    let not_null: i64 = sqlx::query_scalar(
        "SELECT `notnull` FROM pragma_table_info('ducklake_snapshot_changes')
         WHERE name = 'changes_made'",
    )
    .fetch_one(pool)
    .await?;
    if not_null == 0 {
        return Ok(());
    }

    let mut tx = pool.begin().await?;
    sqlx::query(
        "CREATE TABLE ducklake_snapshot_changes__migrate (
            snapshot_id INTEGER PRIMARY KEY,
            changes_made VARCHAR,
            author VARCHAR,
            commit_message VARCHAR,
            commit_extra_info VARCHAR
        )",
    )
    .execute(&mut *tx)
    .await?;
    sqlx::query(
        "INSERT INTO ducklake_snapshot_changes__migrate
             (snapshot_id, changes_made, author, commit_message, commit_extra_info)
         SELECT snapshot_id, NULLIF(changes_made, ''), author, commit_message, commit_extra_info
         FROM ducklake_snapshot_changes",
    )
    .execute(&mut *tx)
    .await?;
    sqlx::query("DROP TABLE ducklake_snapshot_changes")
        .execute(&mut *tx)
        .await?;
    sqlx::query(
        "ALTER TABLE ducklake_snapshot_changes__migrate RENAME TO ducklake_snapshot_changes",
    )
    .execute(&mut *tx)
    .await?;
    tx.commit().await?;
    Ok(())
}

/// Optimistic-concurrency check for a `Replace` commit (mirrors the Postgres
/// writer). Run while holding the SQLite write lock, before retiring the prior
/// generation: if any data file or delete file of the table has `begin_snapshot`
/// or `end_snapshot` newer than `base_snapshot` (the head observed when this
/// write began), or an inlined data or inlined delete row was committed after
/// it, another writer changed the table in the meantime, so this `Replace`
/// aborts with [`DuckLakeError::Conflict`] rather than clobbering it.
/// (`Append` does not call this unless the write set an expected base snapshot:
/// concurrent appends commute.)
/// An inlined-data flush ([`ConflictScope::InlinedFlush`]) does not count delete
/// files: official DuckLake commits a flush over a concurrent delete file.
async fn detect_replace_conflict(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
    base_snapshot: i64,
    scope: ConflictScope,
) -> Result<()> {
    let conflict: Option<i64> = sqlx::query_scalar(
        "SELECT 1 WHERE EXISTS (SELECT 1 FROM ducklake_data_file
             WHERE table_id = ? AND (begin_snapshot > ? OR end_snapshot > ?))
           OR (? AND EXISTS (SELECT 1 FROM ducklake_delete_file
             WHERE table_id = ? AND (begin_snapshot > ? OR end_snapshot > ?)))",
    )
    .bind(table_id)
    .bind(base_snapshot)
    .bind(base_snapshot)
    .bind(scope.counts_delete_files())
    .bind(table_id)
    .bind(base_snapshot)
    .bind(base_snapshot)
    .fetch_optional(&mut **tx)
    .await?;
    if conflict.is_some() {
        return Err(crate::DuckLakeError::Conflict(format!(
            "Replace on table {table_id} conflicts with a concurrent write committed since \
             snapshot {base_snapshot}; aborting (retry the write against the new generation)"
        )));
    }
    for (table_name, system) in inlined_tables(tx, table_id).await? {
        let sql = format!(
            "SELECT 1 FROM {} WHERE {} > ? OR {} > ? LIMIT 1",
            quote_ident(&table_name),
            system.begin_snapshot,
            system.end_snapshot,
        );
        let conflict: Option<i64> = sqlx::query_scalar(AssertSqlSafe(sql))
            .bind(base_snapshot)
            .bind(base_snapshot)
            .fetch_optional(&mut **tx)
            .await?;
        if conflict.is_some() {
            return Err(crate::DuckLakeError::Conflict(format!(
                "Replace on table {table_id} conflicts with inlined data committed since \
                 snapshot {base_snapshot}; aborting"
            )));
        }
    }
    let inlined_deletes = crate::metadata_provider::inlined_delete_table_name(table_id)?;
    let exists: Option<i64> =
        sqlx::query_scalar("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?")
            .bind(&inlined_deletes)
            .fetch_optional(&mut **tx)
            .await?;
    if exists.is_some() {
        let sql = format!(
            "SELECT 1 FROM {} WHERE begin_snapshot > ? LIMIT 1",
            quote_ident(&inlined_deletes)
        );
        let conflict: Option<i64> = sqlx::query_scalar(AssertSqlSafe(sql))
            .bind(base_snapshot)
            .fetch_optional(&mut **tx)
            .await?;
        if conflict.is_some() {
            return Err(crate::DuckLakeError::Conflict(format!(
                "Replace on table {table_id} conflicts with an inlined delete committed since \
                 snapshot {base_snapshot}; aborting"
            )));
        }
    }
    Ok(())
}

// The caller holds SQLite's write lock, so the source-file check is ordered
// with publication.
async fn detect_new_inlined_deletes(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
    base_snapshot: i64,
    sources: &[SourceFile<'_>],
) -> Result<()> {
    if sources.is_empty() {
        return Ok(());
    }

    let table = crate::metadata_provider::inlined_delete_table_name(table_id)?;
    let exists: Option<i64> =
        sqlx::query_scalar("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?")
            .bind(&table)
            .fetch_optional(&mut **tx)
            .await?;
    if exists.is_none() {
        return Ok(());
    }

    let concurrent: Vec<(i64, Option<i64>)> = sqlx::query_as(AssertSqlSafe(format!(
        "SELECT newer.file_id, newer.row_id FROM \"{table}\" newer
         WHERE newer.file_id IN ({})
           AND (newer.begin_snapshot IS NULL OR (
             newer.begin_snapshot > ?
             AND (newer.row_id IS NULL OR NOT EXISTS (
               SELECT 1 FROM \"{table}\" prior
               WHERE prior.file_id = newer.file_id
                 AND prior.row_id = newer.row_id
                 AND prior.begin_snapshot <= ?
             ))
           ))",
        source_file_id_list(sources),
    )))
    .bind(base_snapshot)
    .bind(base_snapshot)
    .fetch_all(&mut **tx)
    .await?;
    inlined_deletion_conflict(table_id, base_snapshot, sources, &concurrent)
}

/// Retire the prior generation's still-visible data files and inlined rows at
/// `snapshot_id`. The `begin_snapshot < snapshot_id` guard spares files
/// registered for *this* snapshot, so a multi-file write does not retire its
/// own siblings. `ducklake_table_stats` is left as it is, as official's
/// `DELETE` leaves it: `record_count` counts every row ever inserted, which is
/// what DuckDB compares with the live count to decide whether the widen-only
/// column bounds are exact. Mirrors the Postgres writer.
async fn retire_prior_generation(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
    snapshot_id: i64,
) -> Result<()> {
    sqlx::query(
        "UPDATE ducklake_data_file SET end_snapshot = ?
         WHERE table_id = ? AND end_snapshot IS NULL AND begin_snapshot < ?",
    )
    .bind(snapshot_id)
    .bind(table_id)
    .bind(snapshot_id)
    .execute(&mut **tx)
    .await?;

    for (table_name, system) in inlined_tables(tx, table_id).await? {
        let sql = format!(
            "UPDATE {} SET {end} = ? WHERE {end} IS NULL AND {begin} < ?",
            quote_ident(&table_name),
            end = system.end_snapshot,
            begin = system.begin_snapshot,
        );
        sqlx::query(AssertSqlSafe(sql))
            .bind(snapshot_id)
            .bind(snapshot_id)
            .execute(&mut **tx)
            .await?;
    }
    Ok(())
}

// The flags report retired Parquet files and inlined rows, respectively
async fn replaced_storage(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
    snapshot_id: i64,
) -> Result<(bool, bool)> {
    let replaced: bool = sqlx::query_scalar(
        "SELECT EXISTS(
            SELECT 1 FROM ducklake_data_file
            WHERE table_id = ? AND end_snapshot = ?
         )",
    )
    .bind(table_id)
    .bind(snapshot_id)
    .fetch_one(&mut **tx)
    .await?;
    for (table_name, system) in inlined_tables(tx, table_id).await? {
        let sql = format!(
            "SELECT EXISTS(SELECT 1 FROM {} WHERE {} = ?)",
            quote_ident(&table_name),
            system.end_snapshot,
        );
        if sqlx::query_scalar(AssertSqlSafe(sql))
            .bind(snapshot_id)
            .fetch_one(&mut **tx)
            .await?
        {
            return Ok((replaced, true));
        }
    }
    Ok((replaced, false))
}

/// Each registered inlined data table of `table_id`, with its system column names.
async fn inlined_tables(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
) -> Result<Vec<(String, InlinedSystemColumns)>> {
    let names: Vec<String> = sqlx::query_scalar(
        "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
    )
    .bind(table_id)
    .fetch_all(&mut **tx)
    .await?;
    let mut tables = Vec::with_capacity(names.len());
    for name in names {
        let system = inlined_table_layout(&mut **tx, &name).await?.system;
        tables.push((name, system));
    }
    Ok(tables)
}

async fn live_inlined_row_count(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
) -> Result<i64> {
    let mut total = 0;
    for (table_name, system) in inlined_tables(tx, table_id).await? {
        let sql = format!(
            "SELECT COUNT(*) FROM {} WHERE {} IS NULL",
            quote_ident(&table_name),
            system.end_snapshot,
        );
        total += sqlx::query_scalar::<_, i64>(AssertSqlSafe(sql))
            .fetch_one(&mut **tx)
            .await?;
    }
    Ok(total)
}

async fn apply_positional_deletes(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
    snapshot_id: i64,
    base_snapshot: i64,
    deletes: &[DeleteFileEntry],
) -> Result<()> {
    for entry in deletes {
        let target_live: Option<i64> = sqlx::query_scalar(
            "SELECT 1 FROM ducklake_data_file
             WHERE data_file_id = ? AND end_snapshot IS NULL",
        )
        .bind(entry.data_file_id)
        .fetch_optional(&mut **tx)
        .await?;
        if target_live.is_none() {
            return Err(crate::DuckLakeError::Conflict(format!(
                "DELETE on data file {} could not commit: the file is no longer live since \
                 snapshot {base_snapshot}",
                entry.data_file_id
            )));
        }
        let current_prev: Option<i64> = sqlx::query_scalar(
            "SELECT delete_file_id FROM ducklake_delete_file
             WHERE data_file_id = ? AND end_snapshot IS NULL",
        )
        .bind(entry.data_file_id)
        .fetch_optional(&mut **tx)
        .await?;
        if current_prev != entry.expected_prev_delete_file {
            return Err(crate::DuckLakeError::Conflict(format!(
                "DELETE on data file {} could not commit: its live delete file changed from \
                 {:?} to {current_prev:?} since snapshot {base_snapshot}",
                entry.data_file_id, entry.expected_prev_delete_file
            )));
        }
        if let Some(previous) = entry.expected_prev_delete_file {
            sqlx::query(
                "UPDATE ducklake_delete_file SET end_snapshot = ?
                 WHERE delete_file_id = ? AND end_snapshot IS NULL",
            )
            .bind(snapshot_id)
            .bind(previous)
            .execute(&mut **tx)
            .await?;
        }
        let file_id = next_file_id(tx).await?;
        sqlx::query(
            "INSERT INTO ducklake_delete_file
                 (delete_file_id, data_file_id, table_id, path, path_is_relative, file_size_bytes,
                  footer_size, delete_count, begin_snapshot)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(file_id)
        .bind(entry.data_file_id)
        .bind(table_id)
        .bind(&entry.delete.path)
        .bind(entry.delete.path_is_relative)
        .bind(entry.delete.file_size_bytes)
        .bind(entry.delete.footer_size)
        .bind(entry.delete.delete_count)
        .bind(snapshot_id)
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

/// Official's table-level conflict check for a staged insert, `DELETE` or
/// `UPDATE` against every snapshot committed after its base snapshot and before
/// `snapshot_id` (see [`staged_conflict`]).
async fn check_staged_conflicts(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    write: &StagedTableWrite,
    snapshot_id: i64,
) -> Result<()> {
    let changes = changes_between(tx, write.base_snapshot_id, snapshot_id).await?;
    staged_conflict(write, &changes).map_or(Ok(()), Err)
}

/// The `changes_made` of every snapshot after `base_snapshot` and before
/// `snapshot_id`, the commit's own.
async fn changes_between(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    base_snapshot: i64,
    snapshot_id: i64,
) -> Result<Vec<Option<String>>> {
    Ok(sqlx::query_scalar(
        "SELECT changes_made FROM ducklake_snapshot_changes
         WHERE snapshot_id > ? AND snapshot_id < ?",
    )
    .bind(base_snapshot)
    .bind(snapshot_id)
    .fetch_all(&mut **tx)
    .await?)
}

/// The table the data files `deletes` delete from belong to: the table the
/// statement read, which a concurrent drop may have replaced under its name.
async fn deleted_files_table(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
    deletes: &[DeleteFileEntry],
) -> Result<i64> {
    let Some(entry) = deletes.first() else {
        return Ok(table_id);
    };
    let source: Option<i64> =
        sqlx::query_scalar("SELECT table_id FROM ducklake_data_file WHERE data_file_id = ?")
            .bind(entry.data_file_id)
            .fetch_optional(&mut **tx)
            .await?;
    Ok(source.unwrap_or(table_id))
}

/// Official's table-level conflict check for a commit at `snapshot_id` making
/// `ours` to `table_id` (see [`table_conflict`]).
async fn check_table_conflicts(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
    base_snapshot: i64,
    snapshot_id: i64,
    ours: TableChanges,
) -> Result<()> {
    let changes = changes_between(tx, base_snapshot, snapshot_id).await?;
    table_conflict(table_id, ours, &changes).map_or(Ok(()), Err)
}

async fn apply_inlined_deletes(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
    snapshot_id: i64,
    base_snapshot: i64,
    rows: &[InlinedRowRef],
) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let registered = inlined_tables(tx, table_id).await?;
    for (name, system) in &registered {
        let sql = format!(
            "SELECT EXISTS(SELECT 1 FROM {} WHERE ({begin} > ? AND {begin} < ?) OR ({end} > ? AND {end} < ?))",
            quote_ident(name),
            begin = system.begin_snapshot,
            end = system.end_snapshot,
        );
        let changed: bool = sqlx::query_scalar(AssertSqlSafe(sql))
            .bind(base_snapshot)
            .bind(snapshot_id)
            .bind(base_snapshot)
            .bind(snapshot_id)
            .fetch_one(&mut **tx)
            .await?;
        if changed {
            return Err(crate::DuckLakeError::Conflict(
                "inlined rows changed since the delete snapshot".to_string(),
            ));
        }
    }
    for (name, row_ids) in inlined_delete_groups(rows) {
        let Some((_, system)) = registered.iter().find(|(table, _)| table == name) else {
            return Err(crate::DuckLakeError::Conflict(format!(
                "inlined deletes reference unregistered table '{name}'"
            )));
        };
        let ids = row_ids
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "UPDATE {} SET {end} = ? WHERE {row_id} IN ({ids}) AND {end} IS NULL AND {begin} <> ?",
            quote_ident(name),
            end = system.end_snapshot,
            row_id = system.row_id,
            begin = system.begin_snapshot,
        );
        sqlx::query(AssertSqlSafe(sql))
            .bind(snapshot_id)
            .bind(snapshot_id)
            .execute(&mut **tx)
            .await?;
    }
    Ok(())
}

/// Record `deletes` in `ducklake_inlined_delete_<table_id>`, creating it as
/// official DuckLake does. Aborts when a target data file is no longer live;
/// the caller has checked the table-level conflicts ([`table_conflict`]).
async fn apply_inlined_file_deletes(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
    snapshot_id: i64,
    base_snapshot: i64,
    deletes: &[InlinedFileDelete],
) -> Result<()> {
    if deletes.is_empty() {
        return Ok(());
    }
    for delete in deletes {
        let live: Option<i64> = sqlx::query_scalar(
            "SELECT 1 FROM ducklake_data_file
             WHERE data_file_id = ? AND table_id = ? AND end_snapshot IS NULL",
        )
        .bind(delete.data_file_id)
        .bind(table_id)
        .fetch_optional(&mut **tx)
        .await?;
        if live.is_none() {
            return Err(crate::DuckLakeError::Conflict(format!(
                "data file {} is no longer live since snapshot {base_snapshot}; re-open the \
                 catalog and retry the statement",
                delete.data_file_id
            )));
        }
        let delete_file: Option<i64> = sqlx::query_scalar(
            "SELECT delete_file_id FROM ducklake_delete_file
             WHERE data_file_id = ? AND end_snapshot IS NULL",
        )
        .bind(delete.data_file_id)
        .fetch_optional(&mut **tx)
        .await?;
        if delete_file != delete.expected_delete_file {
            return Err(crate::metadata_writer::changed_delete_file_conflict(
                delete.data_file_id,
                base_snapshot,
            ));
        }
    }
    let table = crate::metadata_provider::inlined_delete_table_name(table_id)?;
    sqlx::query(AssertSqlSafe(format!(
        "CREATE TABLE IF NOT EXISTS {} (file_id BIGINT, row_id BIGINT, begin_snapshot BIGINT)",
        quote_ident(&table)
    )))
    .execute(&mut **tx)
    .await?;
    let rows = deletes.iter().flat_map(|delete| {
        delete
            .positions
            .iter()
            .map(move |position| (delete.data_file_id, *position))
    });
    let rows = rows.collect::<Vec<_>>();
    for chunk in rows.chunks(300) {
        let mut query = QueryBuilder::<Sqlite>::new(format!(
            "INSERT INTO {} (file_id, row_id, begin_snapshot) ",
            quote_ident(&table)
        ));
        query.push_values(chunk, |mut row, (file_id, position)| {
            row.push_bind(*file_id)
                .push_bind(*position)
                .push_bind(snapshot_id);
        });
        query.build().execute(&mut **tx).await?;
    }
    Ok(())
}

/// End each of `files` at `snapshot_id`, leaving its live delete file as it
/// is, as official's `DropDataFiles` does. Aborts when a file is no longer
/// live or its live delete file changed.
async fn drop_data_files(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
    snapshot_id: i64,
    base_snapshot: i64,
    files: &[DroppedDataFile],
) -> Result<()> {
    for file in files {
        let current: Option<i64> = sqlx::query_scalar(
            "SELECT delete_file_id FROM ducklake_delete_file
             WHERE data_file_id = ? AND end_snapshot IS NULL",
        )
        .bind(file.data_file_id)
        .fetch_optional(&mut **tx)
        .await?;
        let ended = sqlx::query(
            "UPDATE ducklake_data_file SET end_snapshot = ?
             WHERE data_file_id = ? AND table_id = ? AND end_snapshot IS NULL",
        )
        .bind(snapshot_id)
        .bind(file.data_file_id)
        .bind(table_id)
        .execute(&mut **tx)
        .await?
        .rows_affected();
        if ended != 1 || current != file.expected_delete_file {
            return Err(crate::DuckLakeError::Conflict(format!(
                "data file {} or its delete file changed since snapshot {base_snapshot}; \
                 re-open the catalog and retry the statement",
                file.data_file_id
            )));
        }
    }
    Ok(())
}

/// Register each of `entries`, a delete file that records its positions'
/// snapshots, against its data file. One that replaces the file's live delete
/// file carries every position of it, so the replaced row is removed and its
/// object scheduled for deletion, as official's `DeleteOverwrittenDeleteFiles`
/// does. Fails with `conflict()` when a data file's `end_snapshot` is not the
/// entry's, or the delete file it replaces (a live file's live one, a retired
/// file's latest one) is not the entry's.
async fn replace_delete_files(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
    entries: &[InlinedFlushFileDelete],
    conflict: impl Fn() -> crate::DuckLakeError,
) -> Result<()> {
    for entry in entries {
        let target_end: Option<Option<i64>> = sqlx::query_scalar(
            "SELECT end_snapshot FROM ducklake_data_file WHERE data_file_id = ?",
        )
        .bind(entry.data_file_id)
        .fetch_optional(&mut **tx)
        .await?;
        // A live file's live delete file, or a retired file's latest one.
        let current: Option<i64> = sqlx::query_scalar(
            "SELECT delete_file_id FROM ducklake_delete_file
             WHERE data_file_id = ? AND (end_snapshot IS NULL OR ?)
             ORDER BY begin_snapshot DESC LIMIT 1",
        )
        .bind(entry.data_file_id)
        .bind(entry.data_file_end.is_some())
        .fetch_optional(&mut **tx)
        .await?;
        if target_end != Some(entry.data_file_end) || current != entry.expected_prev_delete_file {
            return Err(conflict());
        }
        // A new id, never the replaced row's: a concurrent statement expecting
        // the replaced file must not take the new one for it.
        let delete_file_id = next_file_id(tx).await?;
        if let Some(previous) = entry.expected_prev_delete_file {
            let dead = sqlx::query(AssertSqlSafe(format!(
                "SELECT df.delete_file_id, {RESOLVED_PATH} AS resolved_path,
                        {REL_FLAG} AS rel
                 FROM ducklake_delete_file df
                 JOIN ducklake_table t ON t.table_id = df.table_id
                 JOIN ducklake_schema s ON s.schema_id = t.schema_id
                 WHERE df.delete_file_id = ?"
            )))
            .bind(previous)
            .fetch_all(&mut **tx)
            .await?;
            schedule_files(tx, dead).await?;
            sqlx::query("DELETE FROM ducklake_delete_file WHERE delete_file_id = ?")
                .bind(previous)
                .execute(&mut **tx)
                .await?;
        }
        insert_flush_delete_file(
            tx,
            table_id,
            Some(delete_file_id),
            entry.data_file_id,
            &entry.delete,
        )
        .await?;
    }
    Ok(())
}

/// Register a delete file an inlined flush wrote for `data_file_id`.
/// `delete_file_id` `None` takes the next file id ([`next_file_id`]).
async fn insert_flush_delete_file(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
    delete_file_id: Option<i64>,
    data_file_id: i64,
    delete: &InlinedFlushDeleteFile,
) -> Result<()> {
    let delete_file_id = match delete_file_id {
        Some(id) => id,
        None => next_file_id(tx).await?,
    };
    sqlx::query(
        "INSERT INTO ducklake_delete_file
             (delete_file_id, data_file_id, table_id, path, path_is_relative, file_size_bytes,
              footer_size, delete_count, begin_snapshot, partial_max, end_snapshot)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(delete_file_id)
    .bind(data_file_id)
    .bind(table_id)
    .bind(&delete.delete.path)
    .bind(delete.delete.path_is_relative)
    .bind(delete.delete.file_size_bytes)
    .bind(delete.delete.footer_size)
    .bind(delete.delete.delete_count)
    .bind(delete.begin_snapshot)
    .bind(delete.partial_max)
    .bind(delete.end_snapshot)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Insert the next `ducklake_snapshot` row in commit order, carrying
/// `schema_version` forward (the pure-data-write default), and return
/// `(snapshot_id, schema_version)`.
///
/// This is the FIRST write of a commit transaction, so the `INSERT ... SELECT
/// MAX+1 RETURNING` takes the SQLite write lock up front — preserving the
/// "write-lock-first" invariant that keeps concurrent writers from deadlocking on
/// a read→write lock upgrade. A DDL commit follows this with
/// [`bump_schema_version`]; mirrors the Postgres writer's insert-then-`UPDATE
/// schema_version` shape.
///
/// Unlike the Postgres writer there is no `prev_max == 0 ⇒ 1` data-write floor:
/// on SQLite the first write to any table is always classified as DDL (its
/// `existing` columns are empty → `is_ddl` → bumps to 1), so a `schema_version`
/// of 0 carried forward by a pure data write is unreachable.
async fn insert_snapshot(tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>) -> Result<(i64, i64)> {
    let row = sqlx::query(
        "INSERT INTO ducklake_snapshot (snapshot_id, snapshot_time, schema_version)
         SELECT COALESCE(MAX(snapshot_id), 0) + 1, CURRENT_TIMESTAMP,
                COALESCE(MAX(schema_version), 0)
         FROM ducklake_snapshot
         RETURNING snapshot_id, schema_version",
    )
    .fetch_one(&mut **tx)
    .await?;
    let snapshot_id = row.try_get(0)?;
    let schema_version = row.try_get(1)?;
    sqlx::query(
        "INSERT INTO ducklake_snapshot_changes (snapshot_id, changes_made)
         VALUES (?, NULL)",
    )
    .bind(snapshot_id)
    .execute(&mut **tx)
    .await?;
    Ok((snapshot_id, schema_version))
}

/// Copy the allocator watermarks of the immediately preceding snapshot onto `snapshot_id`. A
/// catalog created by the DuckDB extension carries `next_catalog_id` and `next_file_id` on every
/// snapshot and reads them unconditionally at the head. This crate does not add those columns to
/// its own catalogs, so this carries them forward only where they already exist. It is not part
/// of `insert_snapshot`: the crate allocates partition and sort ids from its own counters and
/// leaves the watermarks NULL, so a later comment must copy that NULL rather than an older value
/// from before the crate allocated its id. A comment allocates no id, so copying is always right.
async fn carry_allocator_watermarks(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    snapshot_id: i64,
) -> Result<()> {
    let allocator_columns: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM pragma_table_info('ducklake_snapshot')
         WHERE name IN ('next_catalog_id', 'next_file_id')",
    )
    .fetch_one(&mut **tx)
    .await?;
    if allocator_columns == 2 {
        sqlx::query(
            "UPDATE ducklake_snapshot
             SET (next_catalog_id, next_file_id) = (
                 SELECT next_catalog_id, next_file_id FROM ducklake_snapshot
                 WHERE snapshot_id < ? ORDER BY snapshot_id DESC LIMIT 1)
             WHERE snapshot_id = ?",
        )
        .bind(snapshot_id)
        .bind(snapshot_id)
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

/// Bump the per-catalog monotonic `schema_version` on a DDL snapshot to
/// `prev_max + 1` (max over the OTHER snapshots, so re-running is stable) and
/// return the new value. Mirrors upstream `if (SchemaChangesMade()) schema_version++`
/// and the Postgres writer's `UPDATE ducklake_snapshot SET schema_version = $1`.
async fn bump_schema_version(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    snapshot_id: i64,
) -> Result<i64> {
    let prev_max: i64 = sqlx::query_scalar(
        "SELECT COALESCE(MAX(schema_version), 0) FROM ducklake_snapshot WHERE snapshot_id <> ?",
    )
    .bind(snapshot_id)
    .fetch_one(&mut **tx)
    .await?;
    let new_version = prev_max + 1;
    sqlx::query("UPDATE ducklake_snapshot SET schema_version = ? WHERE snapshot_id = ?")
        .bind(new_version)
        .bind(snapshot_id)
        .execute(&mut **tx)
        .await?;
    Ok(new_version)
}

/// Record a `ducklake_schema_versions` ledger row for a DDL that leaves the table
/// live (create, column add/remove/reorder, type promotion). Not called for a
/// drop — the table has no live schema afterward (matches upstream `DropTables`
/// and `multicatalog.rs::drop_table_in_catalog`).
async fn record_schema_version(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    snapshot_id: i64,
    schema_version: i64,
    table_id: i64,
) -> Result<()> {
    sqlx::query(
        "INSERT INTO ducklake_schema_versions (begin_snapshot, schema_version, table_id)
         VALUES (?, ?, ?)",
    )
    .bind(snapshot_id)
    .bind(schema_version)
    .bind(table_id)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// The atomic commit point for a single-catalog write. Inserts the deferred
/// `ducklake_snapshot` row (its reserved id was returned by
/// `begin_write_transaction` but NOT inserted there), finalizes the column
/// generation, and — for `Replace` — retires the prior data generation. All
/// within the caller's transaction, so `COALESCE(MAX(snapshot_id), 0)` only
/// ever resolves to a fully-populated head (no transient empty read).
///
/// The column generation is deferred to here (rather than written in
/// `begin_write_transaction` like the Postgres backend) because the SQLite read
/// path resolves a table's columns by `end_snapshot IS NULL` ONLY (not
/// snapshot-scoped), so inserting the new generation at begin would leak it to
/// concurrent reads during the upload window. `column_ids` are the ids reserved
/// at begin and already baked into the staged parquet's `field_id` metadata.
/// Persist a partitioned data file's partition metadata within the commit
/// transaction: set `ducklake_data_file.partition_id` and insert one
/// `ducklake_file_partition_value` row per partition key. A no-op for an
/// unpartitioned file (`partition_id` unset and no values), so every commit path
/// can call it unconditionally.
async fn insert_partition_metadata(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
    data_file_id: i64,
    file: &DataFileInfo,
) -> Result<()> {
    if let Some(partition_id) = file.partition_id {
        sqlx::query("UPDATE ducklake_data_file SET partition_id = ? WHERE data_file_id = ?")
            .bind(partition_id)
            .bind(data_file_id)
            .execute(&mut **tx)
            .await?;
    }
    for (key_index, value) in &file.partition_values {
        sqlx::query(
            "INSERT INTO ducklake_file_partition_value
                 (data_file_id, table_id, partition_key_index, partition_value)
             VALUES (?, ?, ?, ?)",
        )
        .bind(data_file_id)
        .bind(table_id)
        .bind(i64::from(*key_index))
        .bind(value.clone())
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

/// Persist the harvested per-column statistics for a just-registered data file.
///
/// Writes one `ducklake_file_column_stats` row per [`ColumnStat`] (the zone maps
/// that drive file pruning), within the caller's commit transaction so the
/// stats land atomically with the `ducklake_data_file` row. `data_file_id` is
/// the id the row was inserted under (never SQLite's `last_insert_rowid()`: a
/// catalog official DuckDB created declares `data_file_id BIGINT PRIMARY KEY`,
/// which is not SQLite's row id). A `None`
/// bound/count is written as SQL `NULL` — spec-safe: a file with NULL stats is
/// never pruned.
async fn insert_file_column_stats(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
    data_file_id: i64,
    column_stats: &[ColumnStat],
) -> Result<()> {
    crate::metadata_writer::reject_duplicate_column_stats(column_stats)?;
    for stat in column_stats {
        sqlx::query(
            "INSERT INTO ducklake_file_column_stats
                 (data_file_id, table_id, column_id, column_size_bytes,
                  value_count, null_count, min_value, max_value, contains_nan, extra_stats)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
        )
        .bind(data_file_id)
        .bind(table_id)
        .bind(stat.column_id)
        .bind(stat.column_size_bytes)
        .bind(stat.value_count)
        .bind(stat.null_count)
        .bind(stat.min_value.as_deref())
        .bind(stat.max_value.as_deref())
        .bind(stat.contains_nan)
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

/// The field ids and DuckLake types of the table's live columns and nested
/// fields, by which the table-wide statistics merge compares bounds.
async fn live_stat_fields(
    table_id: i64,
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
) -> Result<Vec<(i64, String)>> {
    Ok(sqlx::query_as(
        "SELECT column_id, column_type FROM ducklake_column
         WHERE table_id = ? AND end_snapshot IS NULL",
    )
    .bind(table_id)
    .fetch_all(&mut **tx)
    .await?)
}

/// Merge the statistics of `files`, newly committed to the table, into its
/// `ducklake_table_column_stats` roll-up, as official DuckLake does when it
/// commits data files (see [`crate::stats_encode::merge_file_column_stats`]).
/// Called before the write advances `ducklake_table_stats`, which says whether
/// the table had statistics before it.
async fn merge_file_table_column_stats(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
    fields: &[(i64, String)],
    files: &[Vec<crate::stats_encode::FileColumnStat>],
) -> Result<()> {
    let initialized = table_stats_initialized(tx, table_id).await?;
    let live: Vec<i64> = fields.iter().map(|(id, _)| *id).collect();
    let existing = load_table_column_stats(tx, table_id).await?;
    let merged =
        crate::stats_encode::merge_file_column_stats(existing, initialized, files, &live, |id| {
            crate::metadata_writer::stat_kind_of(fields, id)
        });
    store_table_column_stats(tx, table_id, merged).await
}

/// The table's stored `ducklake_table_column_stats` rows, less those whose
/// bounds are unknown. A row with neither bound reads as holding no valid value
/// (see [`crate::stats_encode::ColumnStats::from_global`]). Official writes one
/// while the column's live files hold only NULLs or only NaN (whose bounds no
/// reader uses once `contains_nan` is set), after a rewrite recomputes the
/// statistics from such files, and for a `GEOMETRY` column, whose bounds are in
/// `extra_stats`; this crate keeps those rows, as official merges into them.
/// Releases of this crate up to 0.9 also wrote one, without NaN, when a live
/// file's bounds were missing: a row with neither bound, no NaN and no
/// `extra_stats` over a live file of an earlier commit not known to hold only
/// NULLs or only NaN is dropped here, so the merge leaves the field with no row
/// (unknown) instead of narrowing it to the new values. The commit's own
/// snapshot is the newest, so its new files are not counted.
async fn load_table_column_stats(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
) -> Result<Vec<crate::stats_encode::GlobalColumnStat>> {
    let mut existing = Vec::new();
    for row in sqlx::query(
        "SELECT s.column_id, s.contains_null, s.contains_nan, s.min_value, s.max_value
         FROM ducklake_table_column_stats s
         WHERE s.table_id = ?
           AND NOT (s.min_value IS NULL AND s.max_value IS NULL
                    AND (s.contains_nan IS NULL OR NOT s.contains_nan)
                    AND s.extra_stats IS NULL AND EXISTS (
               SELECT 1 FROM ducklake_data_file d
               WHERE d.table_id = s.table_id AND d.end_snapshot IS NULL
                 AND d.begin_snapshot < (SELECT MAX(snapshot_id) FROM ducklake_snapshot)
                 AND NOT EXISTS (
                     SELECT 1 FROM ducklake_file_column_stats f
                     WHERE f.data_file_id = d.data_file_id AND f.column_id = s.column_id
                       AND ((f.null_count IS NOT NULL AND f.value_count = 0)
                            OR (f.contains_nan AND f.min_value IS NULL
                                AND f.max_value IS NULL)))))",
    )
    .bind(table_id)
    .fetch_all(&mut **tx)
    .await?
    {
        existing.push(crate::stats_encode::GlobalColumnStat {
            column_id: row.try_get(0)?,
            contains_null: row.try_get(1)?,
            contains_nan: row.try_get(2)?,
            min_value: row.try_get(3)?,
            max_value: row.try_get(4)?,
        });
    }
    Ok(existing)
}

/// Replace the table's `ducklake_table_column_stats` rows with `globals`.
async fn store_table_column_stats(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
    globals: Vec<crate::stats_encode::GlobalColumnStat>,
) -> Result<()> {
    // A kept row keeps its `extra_stats` (a `GEOMETRY` column's bounds), which
    // this crate does not compute.
    let extra: std::collections::HashMap<i64, String> = sqlx::query_as(
        "SELECT column_id, extra_stats FROM ducklake_table_column_stats
         WHERE table_id = ? AND extra_stats IS NOT NULL",
    )
    .bind(table_id)
    .fetch_all(&mut **tx)
    .await?
    .into_iter()
    .collect();
    sqlx::query("DELETE FROM ducklake_table_column_stats WHERE table_id = ?")
        .bind(table_id)
        .execute(&mut **tx)
        .await?;
    for g in globals {
        sqlx::query(
            "INSERT INTO ducklake_table_column_stats
                 (table_id, column_id, contains_null, contains_nan, min_value, max_value, extra_stats)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(table_id)
        .bind(g.column_id)
        .bind(g.contains_null)
        .bind(g.contains_nan)
        .bind(g.min_value)
        .bind(g.max_value)
        .bind(extra.get(&g.column_id))
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

/// Whether the table's statistics were initialized before this write, as
/// official keys an inlined write's statistics on an existing
/// `ducklake_table_stats` row. This crate's commits insert a missing row with
/// zeros before updating it (`INSERT OR IGNORE`), so a zero row may predate
/// any write; a row counts once it has recorded a row (`next_row_id > 0`) or
/// the table has column statistics.
async fn table_stats_initialized(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
) -> Result<bool> {
    Ok(sqlx::query_scalar(
        "SELECT EXISTS (SELECT 1 FROM ducklake_table_stats WHERE table_id = ? AND next_row_id > 0)
             OR EXISTS (SELECT 1 FROM ducklake_table_column_stats WHERE table_id = ?)",
    )
    .bind(table_id)
    .bind(table_id)
    .fetch_one(&mut **tx)
    .await?)
}

/// Merge the statistics of `batches`, written inline to the table, into its
/// `ducklake_table_column_stats`, as official DuckLake does on an inlined write
/// (see [`crate::stats_encode::merge_inlined_column_stats`]). A NULL in one of
/// `snapshot_id_columns` is stored as `snapshot_id`.
#[expect(
    clippy::too_many_arguments,
    reason = "the statistics merge needs the table, its fields, the rows and the stats state"
)]
async fn merge_inlined_table_column_stats(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
    columns: &[ColumnDef],
    column_ids: &[i64],
    batches: &[RecordBatch],
    snapshot_id_columns: &[String],
    snapshot_id: i64,
    initialized: bool,
) -> Result<()> {
    let fields = crate::metadata_writer::catalog_fields(columns, column_ids)?;
    let inlined = crate::stats_encode::inlined_column_stats(batches, &fields, |i| {
        columns
            .get(i)
            .filter(|column| snapshot_id_columns.iter().any(|name| name == column.name()))
            .map(|_| datafusion::common::ScalarValue::Int64(Some(snapshot_id)))
    })?;
    let existing = load_table_column_stats(tx, table_id).await?;
    let merged =
        crate::stats_encode::merge_inlined_column_stats(existing, inlined, initialized, |id| {
            crate::metadata_writer::stat_kind_of(&fields, id)
        });
    store_table_column_stats(tx, table_id, merged).await
}

#[expect(
    clippy::too_many_arguments,
    reason = "snapshot finalization requires the complete atomic write state"
)]
async fn finalize_snapshot(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
    schema_name: &str,
    table_name: &str,
    columns: &[ColumnDef],
    column_ids: &[i64],
    mode: WriteMode,
    base_snapshot: i64,
) -> Result<i64> {
    // One write-lock-first snapshot is shared by every table finalized by a
    // multi-table transaction. A single-table write uses the same path once.
    let (snapshot_id, mut schema_version) = insert_snapshot(tx).await?;
    let mut schema_changed = false;
    finalize_table_snapshot(
        tx,
        snapshot_id,
        &mut schema_version,
        &mut schema_changed,
        table_id,
        schema_name,
        table_name,
        columns,
        column_ids,
        mode,
        base_snapshot,
        true,
    )
    .await?;
    Ok(snapshot_id)
}

#[expect(
    clippy::too_many_arguments,
    reason = "table finalization requires the complete atomic write state"
)]
async fn finalize_table_snapshot(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    snapshot_id: i64,
    schema_version: &mut i64,
    schema_changed: &mut bool,
    table_id: i64,
    schema_name: &str,
    table_name: &str,
    columns: &[ColumnDef],
    column_ids: &[i64],
    mode: WriteMode,
    base_snapshot: i64,
    inserts: bool,
) -> Result<()> {
    // Classify this commit as DDL vs pure data write. The table's begin snapshot
    // identifies creation even when it has no columns; using an empty live-column
    // set would misclassify a later schemaless Replace as another create.
    let (schema_begin_snapshot, table_begin_snapshot): (i64, i64) = sqlx::query_as(
        "SELECT schema.begin_snapshot, table_meta.begin_snapshot
         FROM ducklake_table table_meta
         JOIN ducklake_schema schema ON schema.schema_id = table_meta.schema_id
         WHERE table_meta.table_id = ?",
    )
    .bind(table_id)
    .fetch_one(&mut **tx)
    .await?;

    // `current` is the table's live columns ordered by `column_order`. Mirrors
    // the Postgres writer's `table_was_created || columns_differ` and upstream
    // `SchemaChangesMade()`.
    use std::collections::{HashMap, HashSet};
    let proposed = catalog_column_defs(columns)?;
    if proposed.len() != column_ids.len() {
        return Err(crate::DuckLakeError::InvalidConfig(format!(
            "column_ids has {} entries for {} catalog column nodes",
            column_ids.len(),
            proposed.len()
        )));
    }
    let current = sqlx::query(
        "SELECT column_id, column_name, column_type, column_order, nulls_allowed, parent_column
         FROM ducklake_column
         WHERE table_id = ? AND end_snapshot IS NULL
         ORDER BY column_order",
    )
    .bind(table_id)
    .fetch_all(&mut **tx)
    .await?;
    let existing_catalog_columns = current
        .iter()
        .map(|row| {
            Ok::<_, sqlx::Error>(ExistingCatalogColumn {
                column_id: row.try_get("column_id")?,
                name: row.try_get("column_name")?,
                ducklake_type: row.try_get("column_type")?,
                parent_column: row.try_get("parent_column")?,
            })
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let existing_nullability = current
        .iter()
        .map(|row| {
            Ok::<_, sqlx::Error>(
                row.try_get::<Option<bool>, _>("nulls_allowed")?
                    .unwrap_or(true),
            )
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let committed_ids = assign_column_ids(&proposed, &existing_catalog_columns, column_ids)?;
    if committed_ids != column_ids {
        return Err(crate::DuckLakeError::Conflict(
            "table columns were created concurrently with different field ids; retry the write"
                .to_string(),
        ));
    }
    let table_was_created = table_begin_snapshot == snapshot_id;
    let is_ddl = table_was_created
        || catalog_columns_differ(
            &existing_catalog_columns,
            &existing_nullability,
            &proposed,
            column_ids,
        );
    if is_ddl {
        reject_new_reserved_columns(tx, table_id, &existing_catalog_columns, &proposed).await?;
    }
    if is_ddl && !*schema_changed {
        // A DDL commit bumps the per-catalog schema_version (the insert above only
        // carried it forward). A pure data write keeps the carried value.
        *schema_version = bump_schema_version(tx, snapshot_id).await?;
        *schema_changed = true;
    }

    // Reconcile the column generation SURGICALLY so each column keeps a stable
    // column_id (== parquet field_id) across writes: end only removed columns,
    // insert only new ones, and leave unchanged columns (and their ids) in place.
    // Re-minting ids every write would orphan the field_ids baked into
    // already-written files, making their rows read back as NULL.
    let proposed_ids = column_ids.iter().copied().collect::<HashSet<_>>();
    let mut current_by_id: HashMap<i64, (i64, bool, String)> = HashMap::new();
    for row in &current {
        let column_id: i64 = row.try_get("column_id")?;
        let order: i64 = row.try_get("column_order")?;
        let nullable: bool = row
            .try_get::<Option<bool>, _>("nulls_allowed")?
            .unwrap_or(true);
        let ducklake_type: String = row.try_get("column_type")?;
        if !proposed_ids.contains(&column_id) {
            sqlx::query(
                "UPDATE ducklake_column SET end_snapshot = ?
                 WHERE table_id = ? AND column_id = ? AND end_snapshot IS NULL",
            )
            .bind(snapshot_id)
            .bind(table_id)
            .bind(column_id)
            .execute(&mut **tx)
            .await?;
        }
        current_by_id.insert(column_id, (order, nullable, ducklake_type));
    }

    for (order, (column, column_id)) in proposed.iter().zip(column_ids).enumerate() {
        let parent_id = column.parent_index.map(|index| column_ids[index]);
        match current_by_id.get(column_id) {
            Some((cur_order, cur_nullable, cur_type)) => {
                let migrate_type = catalog_column_type_requires_migration(cur_type, column);
                if migrate_type {
                    sqlx::query(
                        "UPDATE ducklake_column SET end_snapshot = ?
                         WHERE table_id = ? AND column_id = ? AND end_snapshot IS NULL",
                    )
                    .bind(snapshot_id)
                    .bind(table_id)
                    .bind(column_id)
                    .execute(&mut **tx)
                    .await?;
                    sqlx::query(
                        "INSERT INTO ducklake_column
                             (column_id, table_id, column_name, column_type, column_order,
                              nulls_allowed, parent_column, begin_snapshot)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    )
                    .bind(column_id)
                    .bind(table_id)
                    .bind(&column.name)
                    .bind(&column.ducklake_type)
                    .bind(order as i64)
                    .bind(column.is_nullable)
                    .bind(parent_id)
                    .bind(snapshot_id)
                    .execute(&mut **tx)
                    .await?;
                } else if *cur_order != order as i64 || *cur_nullable != column.is_nullable {
                    sqlx::query(
                        "UPDATE ducklake_column
                         SET column_order = ?, nulls_allowed = ?
                         WHERE table_id = ? AND column_id = ? AND end_snapshot IS NULL",
                    )
                    .bind(order as i64)
                    .bind(column.is_nullable)
                    .bind(table_id)
                    .bind(column_id)
                    .execute(&mut **tx)
                    .await?;
                }
            },
            None => {
                sqlx::query(
                    "INSERT INTO ducklake_column
                          (column_id, table_id, column_name, column_type, column_order,
                           nulls_allowed, parent_column, begin_snapshot, initial_default,
                           default_value, default_value_type, default_value_dialect)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                )
                .bind(column_id)
                .bind(table_id)
                .bind(&column.name)
                .bind(&column.ducklake_type)
                .bind(order as i64)
                .bind(column.is_nullable)
                .bind(parent_id)
                .bind(snapshot_id)
                .bind(&column.initial_default)
                .bind(&column.default_value)
                .bind(&column.default_value_type)
                .bind(&column.default_value_dialect)
                .execute(&mut **tx)
                .await?;
            },
        }
    }

    if mode == WriteMode::Replace {
        // Abort if a concurrent writer published a newer generation since this
        // write began (held under the write lock acquired by the MAX+1 insert
        // above, so the check sees a consistent committed state).
        detect_replace_conflict(tx, table_id, base_snapshot, ConflictScope::Table).await?;
        // Retire the prior data generation. A table gets its stats row with
        // its first rows, as official's `CREATE TABLE` writes none.
        retire_prior_generation(tx, table_id, snapshot_id).await?;
    } else if inserts {
        // An insert refuses a drop, alter or delete of the table since its
        // base snapshot, as official's does; a stage that only deletes is
        // checked by its own rules.
        check_table_conflicts(
            tx,
            table_id,
            base_snapshot,
            snapshot_id,
            TableChanges::append(),
        )
        .await?;
    }

    // Record the schema-change ledger row for a DDL commit (table create, or a
    // column add / remove / reorder). A pure data write carries schema_version
    // forward and writes no row. (Drop is handled in `drop_table`: it bumps but
    // writes no row, since the table has no live schema afterward.)
    if is_ddl {
        record_schema_version(tx, snapshot_id, *schema_version, table_id).await?;
        create_eager_inlined_table(tx, table_id, *schema_version).await?;
    }
    let mut ddl_changes = Vec::new();
    if table_was_created {
        if schema_begin_snapshot == table_begin_snapshot {
            // A multi-table commit finalizes each staged table in turn; two
            // tables born with a fresh schema share its begin snapshot, and the
            // schema's creation must reach the ledger exactly once.
            let schema_entry = format!(
                "created_schema:{}",
                crate::metadata_writer::quote_snapshot_name(schema_name),
            );
            let recorded: Option<Option<String>> = sqlx::query_scalar(
                "SELECT changes_made FROM ducklake_snapshot_changes WHERE snapshot_id = ?",
            )
            .bind(snapshot_id)
            .fetch_optional(&mut **tx)
            .await?;
            let duplicate = matches!(
                recorded,
                Some(Some(ref changes)) if snapshot_has_change(changes, &schema_entry)
            );
            if !duplicate {
                ddl_changes.push(schema_entry);
            }
        }
        ddl_changes.push(format!(
            "created_table:{}",
            crate::metadata_writer::quote_snapshot_table(schema_name, table_name),
        ));
    } else if is_ddl {
        ddl_changes.push(format!("altered_table:{table_id}"));
    }
    if !ddl_changes.is_empty() {
        record_snapshot_changes(
            tx,
            snapshot_id,
            &ddl_changes.join(","),
            &SnapshotCommitMetadata::default(),
        )
        .await?;
    }
    Ok(())
}

async fn record_snapshot_changes(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    snapshot_id: i64,
    changes_made: &str,
    commit_metadata: &SnapshotCommitMetadata,
) -> Result<()> {
    let changes_made = (!changes_made.is_empty()).then_some(changes_made);
    sqlx::query(
        "UPDATE ducklake_snapshot_changes
         SET changes_made = CASE
                 WHEN changes_made IS NULL THEN ?
                 WHEN ? IS NULL THEN changes_made
                 ELSE changes_made || ',' || ?
             END,
             author = ?, commit_message = ?, commit_extra_info = ?
         WHERE snapshot_id = ?",
    )
    .bind(changes_made)
    .bind(changes_made)
    .bind(changes_made)
    .bind(commit_metadata.author())
    .bind(commit_metadata.message())
    .bind(commit_metadata.extra_info())
    .bind(snapshot_id)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

async fn validate_staged_table(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    write: &StagedTableWrite,
) -> Result<()> {
    let ended: Option<Option<i64>> =
        sqlx::query_scalar("SELECT end_snapshot FROM ducklake_table WHERE table_id = ?")
            .bind(write.table_id)
            .fetch_optional(&mut **tx)
            .await?;
    if ended.is_none() || matches!(ended, Some(Some(_))) {
        return Err(crate::DuckLakeError::Conflict(
            "staged table was dropped".to_string(),
        ));
    }
    let current = sqlx::query("SELECT column_id, column_type, nulls_allowed, parent_column FROM ducklake_column WHERE table_id = ? AND end_snapshot IS NULL ORDER BY column_order").bind(write.table_id).fetch_all(&mut **tx).await?;
    let proposed = catalog_column_defs(&write.columns)?;
    if current.is_empty() && proposed.len() == write.column_ids.len() {
        return Ok(());
    }
    let mut matches = current.len() == proposed.len() && proposed.len() == write.column_ids.len();
    if matches {
        for (index, (row, column)) in current.iter().zip(&proposed).enumerate() {
            let column_id: i64 = row.try_get("column_id")?;
            let column_type: String = row.try_get("column_type")?;
            let nullable: Option<bool> = row.try_get("nulls_allowed")?;
            let parent: Option<i64> = row.try_get("parent_column")?;
            matches &= column_id == write.column_ids[index]
                && catalog_column_type_equal(&column_type, column)
                && nullable.unwrap_or(true) == column.is_nullable
                && parent == column.parent_index.map(|i| write.column_ids[i]);
        }
    }
    if !matches {
        return Err(crate::DuckLakeError::Conflict(
            "table schema changed after staging".to_string(),
        ));
    }
    Ok(())
}

async fn commit_files_at_snapshot(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    snapshot_id: i64,
    write: &StagedTableWrite,
    files: &[DataFileInfo],
) -> Result<bool> {
    if files.is_empty() {
        return Err(crate::DuckLakeError::InvalidConfig(
            "multi-table file stage must contain at least one file".to_string(),
        ));
    }
    let live_partition_id: Option<i64> = sqlx::query_scalar(
        "SELECT partition_id FROM ducklake_partition_info
         WHERE table_id = ? AND end_snapshot IS NULL",
    )
    .bind(write.table_id)
    .fetch_optional(&mut **tx)
    .await?;
    for file in files {
        crate::metadata_writer::enforce_partition_fence(write.table_id, live_partition_id, file)?;
    }
    sqlx::query(
        "INSERT OR IGNORE INTO ducklake_table_stats
             (table_id, record_count, next_row_id, file_size_bytes)
         VALUES (?, 0, 0, 0)",
    )
    .bind(write.table_id)
    .execute(&mut **tx)
    .await?;
    let mut next_row_id: i64 =
        sqlx::query_scalar("SELECT next_row_id FROM ducklake_table_stats WHERE table_id = ?")
            .bind(write.table_id)
            .fetch_one(&mut **tx)
            .await?;
    let mut total_advance = 0;
    let mut total_records = 0;
    let mut total_bytes = 0;
    for file in files {
        let row_ids = crate::metadata_writer::allocate_row_id_start(next_row_id, file);
        let file_id = next_file_id(tx).await?;
        sqlx::query(
            "INSERT INTO ducklake_data_file
                 (data_file_id, table_id, path, path_is_relative, file_size_bytes, footer_size,
                  record_count, row_id_start, begin_snapshot)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(file_id)
        .bind(write.table_id)
        .bind(&file.path)
        .bind(file.path_is_relative)
        .bind(file.file_size_bytes)
        .bind(file.footer_size)
        .bind(file.record_count)
        .bind(row_ids.stored)
        .bind(snapshot_id)
        .execute(&mut **tx)
        .await?;
        let data_file_id = file_id;
        insert_file_column_stats(tx, write.table_id, data_file_id, &file.column_stats).await?;
        insert_partition_metadata(tx, write.table_id, data_file_id, file).await?;
        next_row_id += row_ids.advance;
        total_advance += row_ids.advance;
        total_records += file.record_count;
        total_bytes += file.file_size_bytes;
    }
    merge_file_table_column_stats(
        tx,
        write.table_id,
        &crate::metadata_writer::catalog_fields(&write.columns, &write.column_ids)?,
        &crate::metadata_writer::file_column_stats(files),
    )
    .await?;
    sqlx::query(
        "UPDATE ducklake_table_stats
         SET next_row_id = next_row_id + ?, record_count = record_count + ?,
             file_size_bytes = file_size_bytes + ?
         WHERE table_id = ?",
    )
    .bind(total_advance)
    .bind(total_records)
    .bind(total_bytes)
    .bind(write.table_id)
    .execute(&mut **tx)
    .await?;
    Ok(sqlx::query_scalar(
        "SELECT EXISTS(
             SELECT 1 FROM ducklake_data_file
             WHERE table_id = ? AND end_snapshot = ?
         )",
    )
    .bind(write.table_id)
    .bind(snapshot_id)
    .fetch_one(&mut **tx)
    .await?)
}

/// The live column rows of `table_id`, each with its parent column id.
async fn live_table_columns(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
) -> Result<Vec<(crate::metadata_provider::DuckLakeTableColumn, Option<i64>)>> {
    let rows = sqlx::query(
        "SELECT column_id, column_name, column_type, nulls_allowed, parent_column
         FROM ducklake_column
         WHERE table_id = ? AND end_snapshot IS NULL
         ORDER BY column_order",
    )
    .bind(table_id)
    .fetch_all(&mut **tx)
    .await?;
    let rows = rows
        .into_iter()
        .map(|row| {
            let nullable: Option<bool> = row.try_get(3)?;
            Ok((
                crate::metadata_provider::DuckLakeTableColumn::new(
                    row.try_get(0)?,
                    row.try_get(1)?,
                    row.try_get(2)?,
                    nullable.unwrap_or(true),
                ),
                row.try_get::<Option<i64>, _>(4)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(rows)
}

/// Create the inlined data table `ducklake_inlined_data_{table_id}_{schema_version}`
/// holding `fields` after the system columns, as official's DDL does, and
/// register it. Returns its name.
async fn create_inlined_data_table(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
    schema_version: i64,
    fields: &[(String, DataType, String)],
) -> Result<String> {
    let physical_name = format!("ducklake_inlined_data_{table_id}_{schema_version}");
    let system = catalog_inlined_system_columns(&mut **tx).await?;
    let mut ddl = format!(
        "CREATE TABLE IF NOT EXISTS {} ({} BIGINT, {} BIGINT, {} BIGINT",
        quote_ident(&physical_name),
        system.row_id,
        system.begin_snapshot,
        system.end_snapshot,
    );
    for (name, data_type, _) in fields {
        ddl.push_str(", ");
        ddl.push_str(&quote_ident(name));
        ddl.push(' ');
        ddl.push_str(inlined_sqlite_type(data_type));
    }
    ddl.push(')');
    sqlx::query(AssertSqlSafe(ddl)).execute(&mut **tx).await?;
    ensure_sqlite_physical_indexes(tx, table_id, &physical_name).await?;
    sqlx::query(
        "INSERT INTO ducklake_inlined_data_tables (table_id, table_name, schema_version)
         SELECT ?, ?, ?
         WHERE NOT EXISTS (
             SELECT 1 FROM ducklake_inlined_data_tables
             WHERE table_id = ? AND schema_version = ?
         )",
    )
    .bind(table_id)
    .bind(&physical_name)
    .bind(schema_version)
    .bind(table_id)
    .bind(schema_version)
    .execute(&mut **tx)
    .await?;
    Ok(physical_name)
}

/// The `data_inlining_row_limit` in effect for the live table `table_id`;
/// `0` when it is not live.
async fn table_inlining_limit(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
) -> Result<usize> {
    let Some(schema_id): Option<i64> = sqlx::query_scalar(
        "SELECT schema_id FROM ducklake_table WHERE table_id = ? AND end_snapshot IS NULL",
    )
    .bind(table_id)
    .fetch_optional(&mut **tx)
    .await?
    else {
        return Ok(0);
    };
    let settings: Vec<(Option<String>, Option<i64>, String)> = sqlx::query_as(
        "SELECT scope, scope_id, value FROM ducklake_metadata
         WHERE key = 'data_inlining_row_limit'",
    )
    .fetch_all(&mut **tx)
    .await?;
    crate::metadata_writer::effective_inlining_limit(&settings, schema_id, table_id)
}

/// Refuse a commit that creates `table_id` or adds a column to it named like
/// an inlined data table's system column while its inlining limit is above
/// `0`, as official refuses `CREATE TABLE` and `ADD COLUMN`.
async fn reject_new_reserved_columns(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
    existing: &[crate::metadata_writer::ExistingCatalogColumn],
    proposed: &[crate::metadata_writer::CatalogColumnDef],
) -> Result<()> {
    let system = catalog_inlined_system_columns(&mut **tx).await?;
    let Some(column) = crate::metadata_writer::new_reserved_column(existing, proposed, system)
    else {
        return Ok(());
    };
    if table_inlining_limit(tx, table_id).await? == 0 {
        return Ok(());
    }
    Err(crate::metadata_writer::reserved_inlining_column_error(
        &column,
    ))
}

/// Official's eager inlined data table: a commit that creates `table_id` or
/// changes its columns or partitioning creates the inlined data table of its
/// `schema_version` while the table's `data_inlining_row_limit` is above `0`
/// and its columns can be inlined (`WriteNewInlinedTables`).
async fn create_eager_inlined_table(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
    schema_version: i64,
) -> Result<()> {
    if table_inlining_limit(tx, table_id).await? == 0 {
        return Ok(());
    }
    let columns = live_table_columns(tx, table_id).await?;
    let system = catalog_inlined_system_columns(&mut **tx).await?;
    if let Some(fields) =
        crate::metadata_writer::inlinable_columns(columns, system, None, sqlite_type_inlines)?
    {
        create_inlined_data_table(tx, table_id, schema_version, &fields).await?;
    }
    Ok(())
}

/// The inlined data table rows of `table_id` written at `snapshot_id` go to:
/// its latest, as official's `LatestInlinedTableQuery` picks it, or, when it
/// has none, a new one for a schema version the commit takes for it, as
/// official's `WriteNewInlinedData` creates one. Official inserts into the
/// latest table whatever its columns; when they are not `fields` the insert
/// fails as official's does, unless this commit itself changed the table's
/// columns (a schema-changing append), which gets a new table.
async fn inlined_insert_table(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
    snapshot_id: i64,
    fields: &[(String, DataType, String)],
) -> Result<String> {
    let latest: Option<String> = sqlx::query_scalar(
        "SELECT table_name FROM ducklake_inlined_data_tables
         WHERE table_id = ? ORDER BY schema_version DESC LIMIT 1",
    )
    .bind(table_id)
    .fetch_optional(&mut **tx)
    .await?;
    if let Some(name) = latest {
        let present: Vec<String> =
            sqlx::query_scalar("SELECT name FROM pragma_table_info(?) ORDER BY cid")
                .bind(&name)
                .fetch_all(&mut **tx)
                .await?;
        let matches = present.len() == fields.len() + 3
            && present[3..]
                .iter()
                .zip(fields)
                .all(|(stored, (name, _, _))| stored == name);
        if matches {
            let borrowed = fields
                .iter()
                .map(|(name, data_type, _)| (name.as_str(), data_type))
                .collect::<Vec<_>>();
            migrate_double_inlined_columns(tx, &name, &borrowed).await?;
            ensure_sqlite_physical_indexes(tx, table_id, &name).await?;
            return Ok(name);
        }
        let changed_now: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM ducklake_column
                           WHERE table_id = ? AND (begin_snapshot = ? OR end_snapshot = ?))",
        )
        .bind(table_id)
        .bind(snapshot_id)
        .bind(snapshot_id)
        .fetch_one(&mut **tx)
        .await?;
        if !changed_now {
            return Err(crate::metadata_writer::inlined_table_shape_error(
                &name,
                present.len(),
                fields.len() + 3,
            ));
        }
    }
    sqlx::query(
        "UPDATE ducklake_snapshot SET schema_version = schema_version + 1 WHERE snapshot_id = ?",
    )
    .bind(snapshot_id)
    .execute(&mut **tx)
    .await?;
    let schema_version: i64 =
        sqlx::query_scalar("SELECT schema_version FROM ducklake_snapshot WHERE snapshot_id = ?")
            .bind(snapshot_id)
            .fetch_one(&mut **tx)
            .await?;
    create_inlined_data_table(tx, table_id, schema_version, fields).await
}

async fn commit_inlined_at_snapshot(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    snapshot_id: i64,
    write: &StagedTableWrite,
    batches: &[RecordBatch],
) -> Result<()> {
    enforce_inline_partition_fence(tx, write.table_id, write.base_snapshot_id).await?;
    let record_count: usize = batches.iter().map(RecordBatch::num_rows).sum();
    if record_count == 0 {
        return Err(crate::DuckLakeError::InvalidConfig(
            "multi-table inline stage must contain at least one row".to_string(),
        ));
    }
    let fields = batches[0]
        .schema()
        .fields()
        .iter()
        .zip(&write.columns)
        .map(|(field, column)| {
            (
                column.name().to_string(),
                field.data_type().clone(),
                column.ducklake_type().to_string(),
            )
        })
        .collect::<Vec<_>>();
    let physical_name = inlined_insert_table(tx, write.table_id, snapshot_id, &fields).await?;
    let system = inlined_table_layout(&mut **tx, &physical_name)
        .await?
        .system;
    let stats_initialized = table_stats_initialized(tx, write.table_id).await?;
    sqlx::query(
        "INSERT OR IGNORE INTO ducklake_table_stats
             (table_id, record_count, next_row_id, file_size_bytes)
         VALUES (?, 0, 0, 0)",
    )
    .bind(write.table_id)
    .execute(&mut **tx)
    .await?;
    let mut row_id: i64 =
        sqlx::query_scalar("SELECT next_row_id FROM ducklake_table_stats WHERE table_id = ?")
            .bind(write.table_id)
            .fetch_one(&mut **tx)
            .await?;
    let column_list = write
        .columns
        .iter()
        .map(|column| quote_ident(column.name()))
        .collect::<Vec<_>>()
        .join(", ");
    let mut written = 0usize;
    for batch in batches {
        for batch_row in 0..batch.num_rows() {
            let mut query = QueryBuilder::<Sqlite>::new(format!(
                "INSERT INTO {} ({}, {}, {}, {}) VALUES (",
                quote_ident(&physical_name),
                system.row_id,
                system.begin_snapshot,
                system.end_snapshot,
                column_list
            ));
            let stored_row_id = match &write.inlined_row_ids {
                Some(row_ids) => row_ids[written],
                None => row_id,
            };
            query.push_bind(stored_row_id);
            query.push(", ").push_bind(snapshot_id);
            query.push(", NULL");
            for (array, column) in batch.columns().iter().zip(&write.columns) {
                query.push(", ");
                if write
                    .snapshot_id_columns
                    .iter()
                    .any(|name| name == column.name())
                    && array.is_null(batch_row)
                {
                    query.push_bind(snapshot_id);
                } else {
                    push_inlined_sqlite_value(&mut query, array.as_ref(), batch_row)?;
                }
            }
            query.push(')');
            query.build().execute(&mut **tx).await?;
            row_id += 1;
            written += 1;
        }
    }
    let record_count = i64::try_from(record_count).map_err(|_| {
        crate::DuckLakeError::InvalidConfig("multi-table inline row count exceeds i64".to_string())
    })?;
    merge_inlined_table_column_stats(
        tx,
        write.table_id,
        &write.columns,
        &write.column_ids,
        batches,
        &write.snapshot_id_columns,
        snapshot_id,
        stats_initialized,
    )
    .await?;
    // Preserved row ids were drawn when the rows were first written.
    let advance = if write.inlined_row_ids.is_some() {
        0
    } else {
        record_count
    };
    sqlx::query(
        "UPDATE ducklake_table_stats
         SET next_row_id = next_row_id + ?, record_count = record_count + ?
         WHERE table_id = ?",
    )
    .bind(advance)
    .bind(record_count)
    .bind(write.table_id)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

impl MetadataWriter for SqliteMetadataWriter {
    fn maintenance(&self) -> Option<&dyn crate::maintenance::CatalogMaintenance> {
        Some(self)
    }

    /// SQLite implements the atomic append-with-deletes commit, so it supports
    /// row-level `UPDATE`.
    fn supports_update(&self) -> bool {
        true
    }

    fn supports_inlined_row_mutations(&self) -> bool {
        true
    }

    fn drop_empty_superseded_inlined_tables(&self) -> Result<()> {
        block_on(async {
            // One transaction: a registry row and its physical table go together.
            let mut tx = self.pool.begin_with("BEGIN IMMEDIATE").await?;
            let targets = sqlx::query_as::<_, (i64, i64, String)>(SUPERSEDED_INLINED_TABLES_SQL)
                .fetch_all(&mut *tx)
                .await?;
            for (table_id, schema_version, table_name) in targets {
                let quoted = quote_ident(&table_name);
                let rows: i64 =
                    sqlx::query_scalar(AssertSqlSafe(format!("SELECT COUNT(*) FROM {quoted}")))
                        .fetch_one(&mut *tx)
                        .await?;
                if rows != 0 {
                    continue;
                }
                sqlx::query(
                    "DELETE FROM ducklake_inlined_data_tables
                     WHERE table_id = ? AND schema_version = ?",
                )
                .bind(table_id)
                .bind(schema_version)
                .execute(&mut *tx)
                .await?;
                sqlx::query(AssertSqlSafe(format!("DROP TABLE IF EXISTS {quoted}")))
                    .execute(&mut *tx)
                    .await?;
            }
            tx.commit().await?;
            Ok(())
        })
    }

    fn set_table_setting(&self, table_id: i64, key: &str, value: &str) -> Result<()> {
        let key = validate_table_setting(key)?;
        block_on(async {
            let mut transaction = self.pool.begin().await?;
            let table_exists: bool = sqlx::query_scalar(
                "SELECT EXISTS(SELECT 1 FROM ducklake_table WHERE table_id = ? AND end_snapshot IS NULL)",
            )
            .bind(table_id)
            .fetch_one(&mut *transaction)
            .await?;
            if !table_exists {
                return Err(crate::DuckLakeError::TableNotFound(table_id.to_string()));
            }
            if enables_inlining(&key, value) {
                let columns: Vec<(String, String)> = sqlx::query_as(SQL_TABLE_INLINING_COLUMNS)
                    .bind(table_id)
                    .fetch_all(&mut *transaction)
                    .await?;
                let system = crate::metadata_provider_sqlite::catalog_inlined_system_columns(
                    &mut *transaction,
                )
                .await?;
                reject_reserved_inlining_columns(&columns, system)?;
            }

            sqlx::query(
                "DELETE FROM ducklake_metadata
                 WHERE key = ? AND scope = 'table' AND scope_id = ?",
            )
            .bind(&key)
            .bind(table_id)
            .execute(&mut *transaction)
            .await?;
            sqlx::query(
                "INSERT INTO ducklake_metadata (key, value, scope, scope_id)
                 VALUES (?, ?, 'table', ?)",
            )
            .bind(&key)
            .bind(value)
            .bind(table_id)
            .execute(&mut *transaction)
            .await?;
            transaction.commit().await?;
            Ok(())
        })
    }

    fn set_inlined_index_columns(&self, table_id: i64, columns: &[String]) -> Result<()> {
        let value = encode_inlined_index_columns(columns)?;
        block_on(async {
            let mut transaction = self.pool.begin().await?;
            let available = sqlx::query_scalar::<_, String>(
                "SELECT column_name FROM ducklake_column
                 WHERE table_id = ? AND end_snapshot IS NULL AND parent_column IS NULL",
            )
            .bind(table_id)
            .fetch_all(&mut *transaction)
            .await?
            .into_iter()
            .collect::<std::collections::HashSet<_>>();
            if available.is_empty() {
                return Err(crate::DuckLakeError::TableNotFound(table_id.to_string()));
            }
            validate_inlined_index_columns(columns, &available)?;
            let existing: Option<String> = sqlx::query_scalar(
                "SELECT value FROM ducklake_metadata
                 WHERE key = ? AND scope = 'table' AND scope_id = ?",
            )
            .bind(INLINED_INDEX_COLUMNS_SETTING)
            .bind(table_id)
            .fetch_optional(&mut *transaction)
            .await?;
            if existing.as_deref() == Some(value.as_str()) {
                transaction.commit().await?;
                return Ok(());
            }
            sqlx::query(
                "DELETE FROM ducklake_metadata
                 WHERE key = ? AND scope = 'table' AND scope_id = ?",
            )
            .bind(INLINED_INDEX_COLUMNS_SETTING)
            .bind(table_id)
            .execute(&mut *transaction)
            .await?;
            sqlx::query(
                "INSERT INTO ducklake_metadata (key, value, scope, scope_id)
                 VALUES (?, ?, 'table', ?)",
            )
            .bind(INLINED_INDEX_COLUMNS_SETTING)
            .bind(value)
            .bind(table_id)
            .execute(&mut *transaction)
            .await?;
            transaction.commit().await?;
            Ok(())
        })
    }

    fn ensure_inlined_indexes(&self, table_id: i64) -> Result<()> {
        block_on(async {
            let mut transaction = self.pool.begin().await?;
            let physical_tables = sqlx::query_scalar::<_, String>(
                "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
            )
            .bind(table_id)
            .fetch_all(&mut *transaction)
            .await?;
            sqlite_inlined_index_columns(&mut transaction, table_id).await?;
            for physical_table in physical_tables {
                ensure_sqlite_physical_indexes(&mut transaction, table_id, &physical_table).await?;
            }
            transaction.commit().await?;
            Ok(())
        })
    }

    fn with_commit_lock(
        &self,
        _identity: &str,
        operation: Box<dyn FnOnce() -> Result<()> + '_>,
    ) -> Result<()> {
        if let Some(lock_path) = &self.lock_path {
            let lock_file = OpenOptions::new()
                .create(true)
                .read(true)
                .truncate(false)
                .write(true)
                .open(lock_path)?;
            File::lock(&lock_file)?;
            let result = operation();
            let unlock = File::unlock(&lock_file);
            return match (result, unlock) {
                (Ok(()), Ok(())) => Ok(()),
                (Ok(()), Err(e)) => Err(e.into()),
                (Err(e), _) => Err(e),
            };
        }

        let _guard = self.commit_lock.lock().map_err(|_| {
            crate::DuckLakeError::Internal("SQLite commit lock is poisoned".to_string())
        })?;
        operation()
    }

    fn catalog_encrypted(&self) -> Result<bool> {
        block_on(async {
            let values: Vec<String> = sqlx::query_scalar(
                "SELECT value FROM ducklake_metadata WHERE key = 'encrypted' AND scope IS NULL",
            )
            .fetch_all(&self.pool)
            .await?;
            crate::metadata_writer::encrypted_option(&values)
        })
    }

    fn head_snapshot(&self) -> Result<Option<i64>> {
        block_on(async {
            Ok(
                sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
                    .fetch_one(&self.pool)
                    .await?,
            )
        })
    }

    fn create_snapshot(&self) -> Result<i64> {
        block_on(async {
            let mut tx = self.pool.begin().await?;
            // A bare snapshot carries no schema change of its own, so carry
            // schema_version forward with the empty change summary inserted by
            // `insert_snapshot`.
            let (snapshot_id, _schema_version) = insert_snapshot(&mut tx).await?;
            tx.commit().await?;
            Ok(snapshot_id)
        })
    }

    fn set_global_setting(&self, key: &str, value: &str) -> Result<()> {
        block_on(async {
            let mut transaction = self.pool.begin().await?;
            if enables_inlining(key, value) {
                let columns: Vec<(String, String)> = sqlx::query_as(SQL_GLOBAL_INLINING_COLUMNS)
                    .fetch_all(&mut *transaction)
                    .await?;
                let system = crate::metadata_provider_sqlite::catalog_inlined_system_columns(
                    &mut *transaction,
                )
                .await?;
                reject_reserved_inlining_columns(&columns, system)?;
            }
            sqlx::query("DELETE FROM ducklake_metadata WHERE key = ? AND scope IS NULL")
                .bind(key)
                .execute(&mut *transaction)
                .await?;
            sqlx::query(
                "INSERT INTO ducklake_metadata (key, value, scope, scope_id) VALUES (?, ?, NULL, NULL)",
            )
            .bind(key)
            .bind(value)
            .execute(&mut *transaction)
            .await?;
            transaction.commit().await?;
            Ok(())
        })
    }

    fn set_tag(&self, target: TagTarget, key: &str, value: Option<&str>) -> Result<i64> {
        validate_name(key, "Tag key")?;
        let change = tag_change(target, key)?;
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let (snapshot_id, _schema_version) = insert_snapshot(&mut tx).await?;
            carry_allocator_watermarks(&mut tx, snapshot_id).await?;
            let live: Option<i64> =
                match target {
                    TagTarget::Object {
                        object_type: TagObjectType::View,
                        object_id,
                    } => sqlx::query_scalar(
                        "SELECT 1 FROM ducklake_view WHERE view_id = ? AND end_snapshot IS NULL",
                    )
                    .bind(object_id)
                    .fetch_optional(&mut *tx)
                    .await?,
                    TagTarget::Object {
                        object_id,
                        ..
                    } => sqlx::query_scalar(
                        "SELECT 1 FROM ducklake_table WHERE table_id = ? AND end_snapshot IS NULL",
                    )
                    .bind(object_id)
                    .fetch_optional(&mut *tx)
                    .await?,
                    TagTarget::Column {
                        table_id,
                        column_id,
                    } => {
                        sqlx::query_scalar(
                            "SELECT 1 FROM ducklake_column
                     WHERE table_id = ? AND column_id = ? AND end_snapshot IS NULL",
                        )
                        .bind(table_id)
                        .bind(column_id)
                        .fetch_optional(&mut *tx)
                        .await?
                    },
                };
            if live.is_none() {
                return Err(crate::DuckLakeError::InvalidConfig(format!(
                    "set_tag: {target:?} is not live in this catalog"
                )));
            }
            // A comment is a schema change in official: it bumps `schema_version`, which
            // attached DuckDB sessions cache the loaded catalog by. Unlike a DDL commit it
            // writes no `ducklake_schema_versions` row.
            bump_schema_version(&mut tx, snapshot_id).await?;
            match target {
                TagTarget::Object {
                    object_id,
                    ..
                } => {
                    sqlx::query(
                        "UPDATE ducklake_tag SET end_snapshot = ?
                         WHERE object_id = ? AND key = ? AND end_snapshot IS NULL",
                    )
                    .bind(snapshot_id)
                    .bind(object_id)
                    .bind(key)
                    .execute(&mut *tx)
                    .await?;
                    sqlx::query(
                        "INSERT INTO ducklake_tag
                             (object_id, begin_snapshot, end_snapshot, key, value)
                         VALUES (?, ?, NULL, ?, ?)",
                    )
                    .bind(object_id)
                    .bind(snapshot_id)
                    .bind(key)
                    .bind(value)
                    .execute(&mut *tx)
                    .await?;
                },
                TagTarget::Column {
                    table_id,
                    column_id,
                } => {
                    sqlx::query(
                        "UPDATE ducklake_column_tag SET end_snapshot = ?
                         WHERE table_id = ? AND column_id = ? AND key = ?
                           AND end_snapshot IS NULL",
                    )
                    .bind(snapshot_id)
                    .bind(table_id)
                    .bind(column_id)
                    .bind(key)
                    .execute(&mut *tx)
                    .await?;
                    sqlx::query(
                        "INSERT INTO ducklake_column_tag
                             (table_id, column_id, begin_snapshot, end_snapshot, key, value)
                         VALUES (?, ?, ?, NULL, ?, ?)",
                    )
                    .bind(table_id)
                    .bind(column_id)
                    .bind(snapshot_id)
                    .bind(key)
                    .bind(value)
                    .execute(&mut *tx)
                    .await?;
                },
            }
            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &change,
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            tx.commit().await?;
            Ok(snapshot_id)
        })
    }

    fn get_or_create_schema(
        &self,
        name: &str,
        path: Option<&str>,
        snapshot_id: i64,
    ) -> Result<(i64, bool)> {
        validate_name(name, "Schema")?;
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let existing = sqlx::query(
                "SELECT schema_id FROM ducklake_schema
                 WHERE schema_name = ? AND end_snapshot IS NULL",
            )
            .bind(name)
            .fetch_optional(&mut *tx)
            .await?;

            if let Some(row) = existing {
                let schema_id = row.try_get(0)?;
                tx.commit().await?;
                return Ok((schema_id, false));
            }

            let schema_path = directory_path(path.unwrap_or(name));
            let row = sqlx::query(
                "INSERT INTO ducklake_schema (schema_name, path, path_is_relative, begin_snapshot)
                 VALUES (?, ?, 1, ?) RETURNING schema_id",
            )
            .bind(name)
            .bind(&schema_path)
            .bind(snapshot_id)
            .fetch_one(&mut *tx)
            .await?;
            let schema_id = row.try_get(0)?;
            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &format!(
                    "created_schema:{}",
                    crate::metadata_writer::quote_snapshot_name(name),
                ),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            tx.commit().await?;
            Ok((schema_id, true))
        })
    }

    fn get_or_create_table(
        &self,
        schema_id: i64,
        name: &str,
        path: Option<&str>,
        snapshot_id: i64,
    ) -> Result<(i64, bool)> {
        validate_name(name, "Table")?;
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let existing = sqlx::query(
                "SELECT table_id FROM ducklake_table
                 WHERE schema_id = ? AND table_name = ? AND end_snapshot IS NULL",
            )
            .bind(schema_id)
            .bind(name)
            .fetch_optional(&mut *tx)
            .await?;

            if let Some(row) = existing {
                let table_id = row.try_get(0)?;
                tx.commit().await?;
                return Ok((table_id, false));
            }

            let schema_name: String =
                sqlx::query_scalar("SELECT schema_name FROM ducklake_schema WHERE schema_id = ?")
                    .bind(schema_id)
                    .fetch_one(&mut *tx)
                    .await?;

            let table_path = directory_path(path.unwrap_or(name));
            let row = sqlx::query(
                "INSERT INTO ducklake_table (schema_id, table_name, path, path_is_relative, begin_snapshot)
                 VALUES (?, ?, ?, 1, ?) RETURNING table_id",
            )
            .bind(schema_id)
            .bind(name)
            .bind(&table_path)
            .bind(snapshot_id)
            .fetch_one(&mut *tx)
            .await?;
            let table_id = row.try_get(0)?;
            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &format!(
                    "created_table:{}",
                    crate::metadata_writer::quote_snapshot_table(&schema_name, name),
                ),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            tx.commit().await?;
            Ok((table_id, true))
        })
    }

    fn promote_column_type(
        &self,
        table_id: i64,
        column_name: &str,
        new_ducklake_type: &str,
    ) -> Result<i64> {
        // Reject an unknown target type up front (before opening a transaction).
        crate::types::ducklake_to_arrow_type(new_ducklake_type)?;
        block_on(async {
            let mut tx = self.pool.begin().await?;

            // Take the write lock up front (write-lock-first invariant; see
            // `insert_snapshot`) by allocating the snapshot before any read, so a
            // concurrent promote/drop blocks on the lock rather than failing a
            // deferred read→write upgrade. If a guard below rejects the promote, the
            // early return drops `tx`, rolling back this snapshot (no trace, no gap).
            let (new_snapshot, _carried) = insert_snapshot(&mut tx).await?;

            // Locate the live version of the column.
            let row = sqlx::query(
                "SELECT column_id, column_type, column_order, nulls_allowed, parent_column,
                        initial_default, default_value, default_value_type, default_value_dialect
                  FROM ducklake_column
                 WHERE table_id = ? AND column_name = ? AND end_snapshot IS NULL
                   AND parent_column IS NULL",
            )
            .bind(table_id)
            .bind(column_name)
            .fetch_optional(&mut *tx)
            .await?
            .ok_or_else(|| {
                crate::DuckLakeError::InvalidConfig(format!(
                    "promote_column_type: no live column '{column_name}' in table {table_id}"
                ))
            })?;
            let column_id: i64 = row.try_get("column_id")?;
            let cur_type: String = row.try_get("column_type")?;
            let column_order: i64 = row.try_get("column_order")?;
            let nulls_allowed: bool = row
                .try_get::<Option<bool>, _>("nulls_allowed")?
                .unwrap_or(true);
            let parent_column: Option<i64> = row.try_get("parent_column")?;
            let initial_default: Option<String> = row.try_get("initial_default")?;
            let default_value: Option<String> = row.try_get("default_value")?;
            let default_value_type: Option<String> = row.try_get("default_value_type")?;
            let default_value_dialect: Option<String> = row.try_get("default_value_dialect")?;

            // No-op / not-a-widening guards. Canonical equality first so an
            // alias-only restatement is reported as "no change", not attempted.
            if crate::types::types_equal_canonical(&cur_type, new_ducklake_type) {
                return Err(crate::DuckLakeError::InvalidConfig(format!(
                    "promote_column_type: column '{column_name}' is already type '{cur_type}' (no change)"
                )));
            }
            if !crate::types::is_promotable(&cur_type, new_ducklake_type) {
                return Err(crate::DuckLakeError::UnsupportedTypeChange {
                    operation: TypeChangeOperation::PromoteColumnType,
                    column: column_name.to_string(),
                    from: cur_type,
                    to: new_ducklake_type.to_string(),
                });
            }

            // A promote IS schema evolution → DDL: bump schema_version on the
            // snapshot allocated above and (below) record the ledger row, matching
            // the Postgres writer and upstream `CHANGE_COLUMN_TYPE`.
            let new_schema_version = bump_schema_version(&mut tx, new_snapshot).await?;

            // Retire the live row and insert a new version with the SAME column_id
            // (stable field-id). Old/new data files each resolve to their snapshot's
            // version; the read path casts old narrow values up to the new type.
            sqlx::query(
                "UPDATE ducklake_column SET end_snapshot = ?
                 WHERE table_id = ? AND column_id = ? AND end_snapshot IS NULL",
            )
            .bind(new_snapshot)
            .bind(table_id)
            .bind(column_id)
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                "INSERT INTO ducklake_column
                     (column_id, begin_snapshot, end_snapshot, table_id, column_order, column_name,
                      column_type, nulls_allowed, parent_column, initial_default, default_value,
                      default_value_type, default_value_dialect)
                 VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(column_id)
            .bind(new_snapshot)
            .bind(table_id)
            .bind(column_order)
            .bind(column_name)
            .bind(new_ducklake_type)
            .bind(nulls_allowed)
            .bind(parent_column)
            .bind(initial_default)
            .bind(default_value)
            .bind(default_value_type)
            .bind(default_value_dialect)
            .execute(&mut *tx)
            .await?;

            // Record the schema-change ledger row so consumers can detect the change.
            record_schema_version(&mut tx, new_snapshot, new_schema_version, table_id).await?;
            create_eager_inlined_table(&mut tx, table_id, new_schema_version).await?;
            record_snapshot_changes(
                &mut tx,
                new_snapshot,
                &format!("altered_table:{table_id}"),
                &SnapshotCommitMetadata::default(),
            )
            .await?;

            // TODO(commit-time type guard, task #4): close the window where an Append
            // whose staging began before this promote commits afterward under the old
            // type (the §5 begin-time reject is the fail-fast layer only). Lower-
            // priority on SQLite than Postgres: SQLite serializes commits on a single
            // write lock (this transaction now holds it from `insert_snapshot` onward),
            // so a promote and a data-write commit can't interleave — only their
            // begin-time staging can, which the §5 reject already guards.

            tx.commit().await?;
            Ok(new_snapshot)
        })
    }

    fn set_partition_spec(
        &self,
        table_id: i64,
        columns: &[(String, PartitionTransform)],
    ) -> Result<i64> {
        if columns.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "set_partition_spec: partition spec must have at least one column; \
                 use reset_partition_spec to remove partitioning"
                    .to_string(),
            ));
        }
        block_on(async {
            let mut tx = self.pool.begin().await?;
            // Write-lock-first: reserve the new partition id (a metadata-counter
            // UPDATE) before any read, so a concurrent DDL blocks on the lock.
            let partition_id = reserve_ids(&mut tx, "next_partition_id", 1).await?;
            let (new_snapshot, _carried) = insert_snapshot(&mut tx).await?;

            // Resolve each partition-key column NAME to its live column_id.
            let mut column_ids: Vec<i64> = Vec::with_capacity(columns.len());
            for (name, _transform) in columns {
                let column_id: i64 = sqlx::query_scalar(
                    "SELECT column_id FROM ducklake_column
                     WHERE table_id = ? AND column_name = ? AND end_snapshot IS NULL
                       AND parent_column IS NULL",
                )
                .bind(table_id)
                .bind(name)
                .fetch_optional(&mut *tx)
                .await?
                .ok_or_else(|| {
                    crate::DuckLakeError::InvalidConfig(format!(
                        "set_partition_spec: no live column '{name}' in table {table_id}"
                    ))
                })?;
                column_ids.push(column_id);
            }

            // End the currently-live spec generation (if any), then insert the new
            // one and its per-key columns.
            sqlx::query(
                "UPDATE ducklake_partition_info SET end_snapshot = ?
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(new_snapshot)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                "INSERT INTO ducklake_partition_info
                     (partition_id, table_id, begin_snapshot, end_snapshot)
                 VALUES (?, ?, ?, NULL)",
            )
            .bind(partition_id)
            .bind(table_id)
            .bind(new_snapshot)
            .execute(&mut *tx)
            .await?;
            for (key_index, column_id) in column_ids.iter().enumerate() {
                sqlx::query(
                    "INSERT INTO ducklake_partition_column
                         (partition_id, table_id, partition_key_index, column_id, transform)
                     VALUES (?, ?, ?, ?, ?)",
                )
                .bind(partition_id)
                .bind(table_id)
                .bind(key_index as i64)
                .bind(*column_id)
                .bind(columns[key_index].1.to_catalog_string())
                .execute(&mut *tx)
                .await?;
            }

            // Setting a spec is DDL: bump + record schema_version.
            let new_schema_version = bump_schema_version(&mut tx, new_snapshot).await?;
            record_schema_version(&mut tx, new_snapshot, new_schema_version, table_id).await?;
            create_eager_inlined_table(&mut tx, table_id, new_schema_version).await?;
            record_snapshot_changes(
                &mut tx,
                new_snapshot,
                &format!("altered_table:{table_id}"),
                &SnapshotCommitMetadata::default(),
            )
            .await?;

            tx.commit().await?;
            Ok(new_snapshot)
        })
    }

    fn live_partition_spec(
        &self,
        table_id: i64,
    ) -> Result<Option<crate::partition::PartitionSpec>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT pi.partition_id, pc.partition_key_index, pc.column_id, pc.transform
                 FROM ducklake_partition_info AS pi
                 JOIN ducklake_partition_column AS pc
                   ON pc.partition_id = pi.partition_id AND pc.table_id = pi.table_id
                 WHERE pi.table_id = ? AND pi.end_snapshot IS NULL
                 ORDER BY pc.partition_key_index",
            )
            .bind(table_id)
            .fetch_all(&self.pool)
            .await?;
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
            // prune_safe = false: this spec is for laying out a write, never pruning.
            Ok(crate::partition::PartitionSpec::from_rows(parsed, false))
        })
    }

    fn reset_partition_spec(&self, table_id: i64) -> Result<i64> {
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let (new_snapshot, _carried) = insert_snapshot(&mut tx).await?;
            let ended = sqlx::query(
                "UPDATE ducklake_partition_info SET end_snapshot = ?
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(new_snapshot)
            .bind(table_id)
            .execute(&mut *tx)
            .await?
            .rows_affected();
            if ended == 0 {
                // Nothing to reset: roll back the just-allocated snapshot (drop tx)
                // so a no-op RESET creates no empty snapshot, and report the head.
                drop(tx);
                let head: i64 = sqlx::query_scalar(
                    "SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot",
                )
                .fetch_one(&self.pool)
                .await?;
                return Ok(head);
            }
            // Removing the spec is DDL: bump + record schema_version.
            let new_schema_version = bump_schema_version(&mut tx, new_snapshot).await?;
            record_schema_version(&mut tx, new_snapshot, new_schema_version, table_id).await?;
            create_eager_inlined_table(&mut tx, table_id, new_schema_version).await?;
            record_snapshot_changes(
                &mut tx,
                new_snapshot,
                &format!("altered_table:{table_id}"),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            tx.commit().await?;
            Ok(new_snapshot)
        })
    }

    fn live_sort_spec(&self, table_id: i64) -> Result<Option<crate::sort::SortSpec>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT si.sort_id, se.sort_key_index, se.expression, se.dialect,
                        se.sort_direction, se.null_order
                 FROM ducklake_sort_info AS si
                 JOIN ducklake_sort_expression AS se
                   ON se.sort_id = si.sort_id AND se.table_id = si.table_id
                 WHERE si.table_id = ? AND si.end_snapshot IS NULL
                 ORDER BY se.sort_key_index",
            )
            .bind(table_id)
            .fetch_all(&self.pool)
            .await?;
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
            Ok(crate::sort::SortSpec::from_rows(parsed))
        })
    }

    fn set_sort_spec(&self, table_id: i64, fields: &[crate::sort::SortField]) -> Result<i64> {
        if fields.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "set_sort_spec: at least one sort key is required (use reset_sort_spec to clear)"
                    .to_string(),
            ));
        }
        block_on(async {
            let mut tx = self.pool.begin().await?;
            // Reserve the sort_id first so the counter UPDATE takes the write lock
            // before any read, so a concurrent DDL blocks on the lock.
            let sort_id = reserve_ids(&mut tx, "next_sort_id", 1).await?;
            let (new_snapshot, _carried) = insert_snapshot(&mut tx).await?;

            // Validate every sort key resolves to a live column (v1 supports bare
            // column keys only), so a bad SET fails here rather than silently
            // producing unsorted writes later.
            for field in fields {
                let column = field.column_candidate().ok_or_else(|| {
                    crate::DuckLakeError::InvalidConfig(format!(
                        "set_sort_spec: sort key '{}' is not a bare column; only column \
                         sort keys are supported",
                        field.expression
                    ))
                })?;
                let exists: Option<i64> = sqlx::query_scalar(
                    "SELECT column_id FROM ducklake_column
                     WHERE table_id = ? AND column_name = ? AND end_snapshot IS NULL
                       AND parent_column IS NULL",
                )
                .bind(table_id)
                .bind(&column)
                .fetch_optional(&mut *tx)
                .await?;
                if exists.is_none() {
                    return Err(crate::DuckLakeError::InvalidConfig(format!(
                        "set_sort_spec: no live column '{column}' in table {table_id}"
                    )));
                }
            }

            // End the currently-live spec generation (if any), then insert the new
            // one and its per-key expressions.
            sqlx::query(
                "UPDATE ducklake_sort_info SET end_snapshot = ?
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(new_snapshot)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                "INSERT INTO ducklake_sort_info
                     (sort_id, table_id, begin_snapshot, end_snapshot)
                 VALUES (?, ?, ?, NULL)",
            )
            .bind(sort_id)
            .bind(table_id)
            .bind(new_snapshot)
            .execute(&mut *tx)
            .await?;
            for field in fields {
                sqlx::query(
                    "INSERT INTO ducklake_sort_expression
                         (sort_id, table_id, sort_key_index, expression, dialect,
                          sort_direction, null_order)
                     VALUES (?, ?, ?, ?, ?, ?, ?)",
                )
                .bind(sort_id)
                .bind(table_id)
                .bind(field.sort_key_index as i64)
                .bind(&field.expression)
                .bind(&field.dialect)
                .bind(field.direction.to_catalog_string())
                .bind(field.null_order.to_catalog_string())
                .execute(&mut *tx)
                .await?;
            }

            // A sort-order change does NOT bump schema_version (see the trait doc):
            // it does not alter the logical schema, only future write layout.
            record_snapshot_changes(
                &mut tx,
                new_snapshot,
                &format!("altered_table:{table_id}"),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            tx.commit().await?;
            Ok(new_snapshot)
        })
    }

    fn reset_sort_spec(&self, table_id: i64) -> Result<i64> {
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let (new_snapshot, _carried) = insert_snapshot(&mut tx).await?;
            let ended = sqlx::query(
                "UPDATE ducklake_sort_info SET end_snapshot = ?
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(new_snapshot)
            .bind(table_id)
            .execute(&mut *tx)
            .await?
            .rows_affected();
            if ended == 0 {
                // Nothing to reset: roll back the just-allocated snapshot (drop tx)
                // so a no-op RESET creates no empty snapshot, and report the head.
                drop(tx);
                let head: i64 = sqlx::query_scalar(
                    "SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot",
                )
                .fetch_one(&self.pool)
                .await?;
                return Ok(head);
            }
            // A sort-order change does NOT bump schema_version.
            record_snapshot_changes(
                &mut tx,
                new_snapshot,
                &format!("altered_table:{table_id}"),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            tx.commit().await?;
            Ok(new_snapshot)
        })
    }

    fn set_columns(
        &self,
        table_id: i64,
        columns: &[ColumnDef],
        snapshot_id: i64,
    ) -> Result<Vec<i64>> {
        if columns.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "Table must have at least one column".to_string(),
            ));
        }
        block_on(async {
            // Use a transaction to ensure atomicity: if column insertion fails,
            // we don't leave existing columns marked as ended
            let mut tx = self.pool.begin().await?;
            let table_begin_snapshot: i64 =
                sqlx::query_scalar("SELECT begin_snapshot FROM ducklake_table WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;

            sqlx::query(
                "UPDATE ducklake_column SET end_snapshot = ?
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(snapshot_id)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            // Reserve a contiguous column_id block from the monotonic counter and
            // insert with explicit ids, keeping the allocator authoritative (the
            // write path's begin/commit reserve column ids from the same counter).
            let catalog_columns = catalog_column_defs(columns)?;
            let n = catalog_columns.len() as i64;
            let last_column_id = reserve_ids(&mut tx, "next_column_id", n).await?;
            let first_column_id = last_column_id - n + 1;
            let column_ids = (first_column_id..=last_column_id).collect::<Vec<_>>();
            for (order, (column, column_id)) in
                catalog_columns.iter().zip(column_ids.iter()).enumerate()
            {
                let parent_id = column.parent_index.map(|index| column_ids[index]);
                sqlx::query(
                    "INSERT INTO ducklake_column
                          (column_id, table_id, column_name, column_type, column_order,
                           nulls_allowed, parent_column, begin_snapshot, initial_default,
                           default_value, default_value_type, default_value_dialect)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                )
                .bind(column_id)
                .bind(table_id)
                .bind(&column.name)
                .bind(&column.ducklake_type)
                .bind(order as i64)
                .bind(column.is_nullable)
                .bind(parent_id)
                .bind(snapshot_id)
                .bind(&column.initial_default)
                .bind(&column.default_value)
                .bind(&column.default_value_type)
                .bind(&column.default_value_dialect)
                .execute(&mut *tx)
                .await?;
            }

            if table_begin_snapshot != snapshot_id {
                record_snapshot_changes(
                    &mut tx,
                    snapshot_id,
                    &format!("altered_table:{table_id}"),
                    &SnapshotCommitMetadata::default(),
                )
                .await?;
            }
            tx.commit().await?;
            top_level_column_ids(&catalog_columns, &column_ids)
        })
    }

    fn register_data_file(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        snapshot_id: i64,
        file: &DataFileInfo,
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
    ) -> Result<CommitIds> {
        self.register_data_file_with_commit_metadata(
            table_id,
            schema_name,
            table_name,
            snapshot_id,
            file,
            mode,
            base_snapshot,
            columns,
            column_ids,
            &SnapshotCommitMetadata::default(),
            None,
        )
    }

    fn register_data_file_with_commit_metadata(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        _snapshot_id: i64,
        file: &DataFileInfo,
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
        commit_metadata: &SnapshotCommitMetadata,
        expected_base_snapshot_id: Option<i64>,
    ) -> Result<CommitIds> {
        block_on(async {
            // Single atomic commit: insert the deferred snapshot row + finalize
            // the column generation + retire the prior generation (Replace),
            // then register this file and advance the monotonic row-lineage
            // counter — all in one transaction, so the head (MAX(snapshot_id))
            // only ever resolves to fully-populated data (no empty-read window).
            let mut tx = self.pool.begin().await?;

            let snapshot_id = finalize_snapshot(
                &mut tx,
                table_id,
                schema_name,
                table_name,
                columns,
                column_ids,
                mode,
                base_snapshot,
            )
            .await?;
            if mode != WriteMode::Replace
                && let Some(expected_base_snapshot_id) = expected_base_snapshot_id
            {
                detect_replace_conflict(
                    &mut tx,
                    table_id,
                    expected_base_snapshot_id,
                    ConflictScope::Table,
                )
                .await?;
            }

            // Partition-spec fence: this file must be consistent with the table's live
            // partition generation at commit time (both directions — see
            // enforce_partition_fence). ducklake_partition_info is created at init, so
            // this query is always valid; the tx rolls back on a Conflict.
            let live_partition_id: Option<i64> = sqlx::query_scalar(
                "SELECT partition_id FROM ducklake_partition_info
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(table_id)
            .fetch_optional(&mut *tx)
            .await?;
            crate::metadata_writer::enforce_partition_fence(table_id, live_partition_id, file)?;

            // Seed the stats row for the Append path (Replace already seeded it
            // in finalize_snapshot); INSERT OR IGNORE is a no-op if it exists.
            sqlx::query(
                "INSERT OR IGNORE INTO ducklake_table_stats
                     (table_id, record_count, next_row_id, file_size_bytes)
                 VALUES (?, 0, 0, 0)",
            )
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            let stats_row =
                sqlx::query("SELECT next_row_id FROM ducklake_table_stats WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;
            let next_row_id: i64 = stats_row.try_get(0)?;
            let row_ids = crate::metadata_writer::allocate_row_id_start(next_row_id, file);

            let file_id = next_file_id(&mut tx).await?;
            sqlx::query(
                "INSERT INTO ducklake_data_file
                     (data_file_id, table_id, path, path_is_relative, file_size_bytes,
                      footer_size, record_count, row_id_start, begin_snapshot)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(file_id)
            .bind(table_id)
            .bind(&file.path)
            .bind(file.path_is_relative)
            .bind(file.file_size_bytes)
            .bind(file.footer_size)
            .bind(file.record_count)
            .bind(row_ids.stored)
            .bind(snapshot_id)
            .execute(&mut *tx)
            .await?;

            // Persist the file's per-column zone-map stats in the same tx.
            let data_file_id = file_id;
            insert_file_column_stats(&mut tx, table_id, data_file_id, &file.column_stats).await?;
            insert_partition_metadata(&mut tx, table_id, data_file_id, file).await?;
            // Merge the file's statistics into the table-wide roll-up.
            merge_file_table_column_stats(
                &mut tx,
                table_id,
                &crate::metadata_writer::catalog_fields(columns, column_ids)?,
                &crate::metadata_writer::file_column_stats([file]),
            )
            .await?;

            // Advance the counter and accumulate stats. `next_row_id`
            // monotonically increases over the table's lifetime — rowids
            // are never reused, even after end-snapshot.
            sqlx::query(
                "UPDATE ducklake_table_stats
                 SET next_row_id     = next_row_id + ?,
                     record_count    = record_count + ?,
                     file_size_bytes = file_size_bytes + ?
                 WHERE table_id = ?",
            )
            .bind(row_ids.advance)
            .bind(file.record_count)
            .bind(file.file_size_bytes)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            let replaced_existing_data = replaced_storage(&mut tx, table_id, snapshot_id).await?;
            let changes_made = table_write_changes(table_id, mode, false, replaced_existing_data);
            record_snapshot_changes(&mut tx, snapshot_id, &changes_made, commit_metadata).await?;

            let schema_id: i64 =
                sqlx::query("SELECT schema_id FROM ducklake_table WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?
                    .try_get(0)?;

            tx.commit().await?;
            Ok(CommitIds {
                snapshot_id,
                schema_id,
                table_id,
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn register_data_files(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        snapshot_id: i64,
        files: &[DataFileInfo],
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
    ) -> Result<CommitIds> {
        self.register_data_files_with_commit_metadata(
            table_id,
            schema_name,
            table_name,
            snapshot_id,
            files,
            mode,
            base_snapshot,
            columns,
            column_ids,
            &SnapshotCommitMetadata::default(),
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn register_data_files_with_commit_metadata(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        _snapshot_id: i64,
        files: &[DataFileInfo],
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
        commit_metadata: &SnapshotCommitMetadata,
        expected_base_snapshot_id: Option<i64>,
    ) -> Result<CommitIds> {
        if files.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "register_data_files: files must be non-empty".to_string(),
            ));
        }
        block_on(async {
            // One atomic snapshot for all N partition files: finalize the snapshot +
            // (Replace) retire the prior generation ONCE, then insert every file with
            // a distinct row_id_start from the advancing row-lineage counter.
            let mut tx = self.pool.begin().await?;
            let snapshot_id = finalize_snapshot(
                &mut tx,
                table_id,
                schema_name,
                table_name,
                columns,
                column_ids,
                mode,
                base_snapshot,
            )
            .await?;
            if mode != WriteMode::Replace
                && let Some(expected_base_snapshot_id) = expected_base_snapshot_id
            {
                detect_replace_conflict(
                    &mut tx,
                    table_id,
                    expected_base_snapshot_id,
                    ConflictScope::Table,
                )
                .await?;
            }
            // Partition-spec fence (both directions, every file): each file must be
            // consistent with the table's live partition generation at commit time —
            // no file stamped with a retired partition_id, and no non-empty
            // partition_id-less file in a partitioned table. The Append path is
            // otherwise unfenced (finalize_snapshot only conflict-checks Replace); the
            // tx (incl. the snapshot just allocated) rolls back on a Conflict.
            let live_partition_id: Option<i64> = sqlx::query_scalar(
                "SELECT partition_id FROM ducklake_partition_info
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(table_id)
            .fetch_optional(&mut *tx)
            .await?;
            for file in files {
                crate::metadata_writer::enforce_partition_fence(table_id, live_partition_id, file)?;
            }
            sqlx::query(
                "INSERT OR IGNORE INTO ducklake_table_stats
                     (table_id, record_count, next_row_id, file_size_bytes)
                 VALUES (?, 0, 0, 0)",
            )
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            let mut next_row_id: i64 =
                sqlx::query("SELECT next_row_id FROM ducklake_table_stats WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?
                    .try_get(0)?;
            let mut total_advance: i64 = 0;
            let mut total_records: i64 = 0;
            let mut total_bytes: i64 = 0;
            for file in files {
                let row_ids = crate::metadata_writer::allocate_row_id_start(next_row_id, file);
                let file_id = next_file_id(&mut tx).await?;
                sqlx::query(
                    "INSERT INTO ducklake_data_file
                         (data_file_id, table_id, path, path_is_relative, file_size_bytes,
                          footer_size, record_count, row_id_start, begin_snapshot)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                )
                .bind(file_id)
                .bind(table_id)
                .bind(&file.path)
                .bind(file.path_is_relative)
                .bind(file.file_size_bytes)
                .bind(file.footer_size)
                .bind(file.record_count)
                .bind(row_ids.stored)
                .bind(snapshot_id)
                .execute(&mut *tx)
                .await?;
                let data_file_id = file_id;
                insert_file_column_stats(&mut tx, table_id, data_file_id, &file.column_stats)
                    .await?;
                insert_partition_metadata(&mut tx, table_id, data_file_id, file).await?;
                next_row_id += row_ids.advance;
                total_advance += row_ids.advance;
                total_records += file.record_count;
                total_bytes += file.file_size_bytes;
            }
            // Merge the files' statistics into the table-wide roll-up.
            merge_file_table_column_stats(
                &mut tx,
                table_id,
                &crate::metadata_writer::catalog_fields(columns, column_ids)?,
                &crate::metadata_writer::file_column_stats(files),
            )
            .await?;
            sqlx::query(
                "UPDATE ducklake_table_stats
                 SET next_row_id     = next_row_id + ?,
                     record_count    = record_count + ?,
                     file_size_bytes = file_size_bytes + ?
                 WHERE table_id = ?",
            )
            .bind(total_advance)
            .bind(total_records)
            .bind(total_bytes)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            let replaced_existing_data = replaced_storage(&mut tx, table_id, snapshot_id).await?;
            let changes_made = table_write_changes(table_id, mode, false, replaced_existing_data);
            record_snapshot_changes(&mut tx, snapshot_id, &changes_made, commit_metadata).await?;
            let schema_id: i64 =
                sqlx::query("SELECT schema_id FROM ducklake_table WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?
                    .try_get(0)?;
            tx.commit().await?;
            Ok(CommitIds {
                snapshot_id,
                schema_id,
                table_id,
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn reserved_inlining_column(&self, columns: &[String]) -> Result<Option<String>> {
        let system = block_on(catalog_inlined_system_columns(&self.pool))?;
        Ok(columns
            .iter()
            .find(|column| system.conflicts_with(column))
            .cloned())
    }

    fn supports_data_inlining(&self, schema: &arrow::datatypes::Schema) -> bool {
        let Ok(system) = block_on(catalog_inlined_system_columns(&self.pool)) else {
            return false;
        };
        if schema
            .fields()
            .iter()
            .any(|field| system.conflicts_with(field.name()))
        {
            return false;
        }
        schema
            .fields()
            .iter()
            .all(|field| sqlite_type_inlines(field.data_type()))
    }

    fn table_types_support_inlining(&self, schema_name: &str, table_name: &str) -> Result<bool> {
        block_on(async {
            let refused: bool = sqlx::query_scalar(AssertSqlSafe(format!(
                "SELECT EXISTS(
                     SELECT 1 FROM ducklake_column c
                     JOIN ducklake_table t ON t.table_id = c.table_id
                     JOIN ducklake_schema s ON s.schema_id = t.schema_id
                     WHERE s.schema_name = ? AND s.end_snapshot IS NULL
                       AND t.table_name = ? AND t.end_snapshot IS NULL
                       AND c.end_snapshot IS NULL AND LOWER(c.column_type) IN ({}))",
                crate::types::sql_uninlinable_types()
            )))
            .bind(schema_name)
            .bind(table_name)
            .fetch_one(&self.pool)
            .await?;
            Ok(!refused)
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn register_inlined_data(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        _snapshot_id: i64,
        batches: &[RecordBatch],
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
        commit_metadata: &SnapshotCommitMetadata,
        expected_base_snapshot_id: Option<i64>,
    ) -> Result<CommitIds> {
        block_on(async {
            let record_count: usize = batches.iter().map(RecordBatch::num_rows).sum();
            if record_count == 0 {
                return Err(crate::DuckLakeError::InvalidConfig(
                    "register_inlined_data: batches must contain at least one row".to_string(),
                ));
            }
            if batches
                .iter()
                .any(|batch| batch.num_columns() != columns.len())
            {
                return Err(crate::DuckLakeError::InvalidConfig(
                    "register_inlined_data: batch schema does not match table columns".to_string(),
                ));
            }

            let mut tx = self.pool.begin().await?;
            enforce_inline_partition_fence(&mut tx, table_id, base_snapshot).await?;
            let snapshot_id = finalize_snapshot(
                &mut tx,
                table_id,
                schema_name,
                table_name,
                columns,
                column_ids,
                mode,
                base_snapshot,
            )
            .await?;
            if mode != WriteMode::Replace
                && let Some(expected_base_snapshot_id) = expected_base_snapshot_id
            {
                detect_replace_conflict(
                    &mut tx,
                    table_id,
                    expected_base_snapshot_id,
                    ConflictScope::Table,
                )
                .await?;
            }
            let fields = batches[0]
                .schema()
                .fields()
                .iter()
                .zip(columns)
                .map(|(field, column)| {
                    (
                        column.name().to_string(),
                        field.data_type().clone(),
                        column.ducklake_type().to_string(),
                    )
                })
                .collect::<Vec<_>>();
            let physical_name =
                inlined_insert_table(&mut tx, table_id, snapshot_id, &fields).await?;
            let system = inlined_table_layout(&mut *tx, &physical_name).await?.system;

            let stats_initialized = table_stats_initialized(&mut tx, table_id).await?;
            sqlx::query(
                "INSERT OR IGNORE INTO ducklake_table_stats
                     (table_id, record_count, next_row_id, file_size_bytes)
                 VALUES (?, 0, 0, 0)",
            )
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            let mut row_id: i64 = sqlx::query_scalar(
                "SELECT next_row_id FROM ducklake_table_stats WHERE table_id = ?",
            )
            .bind(table_id)
            .fetch_one(&mut *tx)
            .await?;

            let column_list = columns
                .iter()
                .map(|column| quote_ident(column.name()))
                .collect::<Vec<_>>()
                .join(", ");
            for batch in batches {
                for batch_row in 0..batch.num_rows() {
                    let mut query = QueryBuilder::<Sqlite>::new(format!(
                        "INSERT INTO {} ({}, {}, {}, {}) VALUES (",
                        quote_ident(&physical_name),
                        system.row_id,
                        system.begin_snapshot,
                        system.end_snapshot,
                        column_list
                    ));
                    query.push_bind(row_id);
                    query.push(", ").push_bind(snapshot_id);
                    query.push(", NULL");
                    for array in batch.columns() {
                        query.push(", ");
                        push_inlined_sqlite_value(&mut query, array.as_ref(), batch_row)?;
                    }
                    query.push(')');
                    query.build().execute(&mut *tx).await?;
                    row_id += 1;
                }
            }

            merge_inlined_table_column_stats(
                &mut tx,
                table_id,
                columns,
                column_ids,
                batches,
                &[],
                snapshot_id,
                stats_initialized,
            )
            .await?;
            let record_count = i64::try_from(record_count).map_err(|_| {
                crate::DuckLakeError::InvalidConfig(
                    "register_inlined_data: record count exceeds i64".to_string(),
                )
            })?;
            sqlx::query(
                "UPDATE ducklake_table_stats
                 SET next_row_id = next_row_id + ?, record_count = record_count + ?
                 WHERE table_id = ?",
            )
            .bind(record_count)
            .bind(record_count)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            // Append to the ledger row finalize_snapshot seeded (which may already
            // carry created_schema:/created_table: entries) instead of replacing
            // it, and record a Replace over prior data — Parquet or inlined — as
            // delete + insert, the same semantics the Parquet path records.
            let (replaced_files, replaced_inlined) =
                replaced_storage(&mut tx, table_id, snapshot_id).await?;
            let changes_made = table_storage_changes(
                table_id,
                Some(true),
                mode == WriteMode::Replace && replaced_files,
                mode == WriteMode::Replace && replaced_inlined,
            );
            record_snapshot_changes(&mut tx, snapshot_id, &changes_made, commit_metadata).await?;

            let schema_id: i64 =
                sqlx::query_scalar("SELECT schema_id FROM ducklake_table WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;
            tx.commit().await?;
            Ok(CommitIds {
                snapshot_id,
                schema_id,
                table_id,
            })
        })
    }

    fn commit_multi_table(
        &self,
        writes: &[StagedTableWrite],
        commit_metadata: &SnapshotCommitMetadata,
        expected_base_snapshot_id: Option<i64>,
    ) -> Result<MultiTableCommit> {
        if writes.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "commit_multi_table requires at least one table stage".to_string(),
            ));
        }
        let table_ids = writes
            .iter()
            .map(|write| write.table_id)
            .collect::<std::collections::HashSet<_>>();
        if table_ids.len() != writes.len() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "commit_multi_table requires one stage per table".to_string(),
            ));
        }

        block_on(async {
            for write in writes {
                validate_row_mutation_stage(write)?;
            }
            let mut tx = self.pool.begin_with("BEGIN IMMEDIATE").await?;
            for write in writes {
                // Official checks the table-level conflicts first.
                check_staged_conflicts(&mut tx, write, i64::MAX).await?;
                validate_staged_table(&mut tx, write).await?;
                detect_new_inlined_deletes(
                    &mut tx,
                    write.table_id,
                    write.base_snapshot_id,
                    &row_mutation_source_files(write),
                )
                .await?;
            }
            if let Some(expected) = expected_base_snapshot_id {
                for write in writes {
                    detect_replace_conflict(
                        &mut tx,
                        write.table_id,
                        expected,
                        ConflictScope::of(write),
                    )
                    .await?;
                }
            }
            let mut had_live_data = Vec::with_capacity(writes.len());
            for write in writes {
                let files: bool = sqlx::query_scalar(
                    "SELECT EXISTS(
                         SELECT 1 FROM ducklake_data_file
                         WHERE table_id = ? AND end_snapshot IS NULL
                     )",
                )
                .bind(write.table_id)
                .fetch_one(&mut *tx)
                .await?;
                had_live_data.push((
                    files,
                    live_inlined_row_count(&mut tx, write.table_id).await? > 0,
                ));
            }
            let (snapshot_id, mut schema_version) = insert_snapshot(&mut tx).await?;
            for write in writes {
                if let Some(dated) = rewritten_deletes_snapshot(write) {
                    let between: bool = sqlx::query_scalar(
                        "SELECT EXISTS(SELECT 1 FROM ducklake_snapshot
                                       WHERE snapshot_id >= ? AND snapshot_id < ?)",
                    )
                    .bind(dated)
                    .bind(snapshot_id)
                    .fetch_one(&mut *tx)
                    .await?;
                    if between {
                        return Err(rewritten_delete_conflict(write.table_id, dated));
                    }
                }
            }
            let mut schema_changed = false;
            for write in writes {
                finalize_table_snapshot(
                    &mut tx,
                    snapshot_id,
                    &mut schema_version,
                    &mut schema_changed,
                    write.table_id,
                    &write.schema_name,
                    &write.table_name,
                    &write.columns,
                    &write.column_ids,
                    write.mode,
                    write.base_snapshot_id,
                    !matches!(write.data, StagedTableData::None),
                )
                .await?;
            }

            let mut tables = Vec::with_capacity(writes.len());
            for (write, (had_files, had_inlined_rows)) in writes.iter().zip(had_live_data) {
                match &write.data {
                    StagedTableData::Files(files) => {
                        commit_files_at_snapshot(&mut tx, snapshot_id, write, files).await?;
                    },
                    StagedTableData::Inlined(batches) => {
                        commit_inlined_at_snapshot(&mut tx, snapshot_id, write, batches).await?;
                    },
                    StagedTableData::None => {},
                }
                apply_positional_deletes(
                    &mut tx,
                    write.table_id,
                    snapshot_id,
                    write.base_snapshot_id,
                    &write.positional_deletes,
                )
                .await?;
                replace_delete_files(&mut tx, write.table_id, &write.rewritten_deletes, || {
                    crate::DuckLakeError::Conflict(format!(
                        "a data file of table {} or its delete file changed since snapshot {}; \
                         re-open the catalog and retry the statement",
                        write.table_id, write.base_snapshot_id
                    ))
                })
                .await?;
                apply_inlined_deletes(
                    &mut tx,
                    write.table_id,
                    snapshot_id,
                    write.base_snapshot_id,
                    &write.inlined_deletes,
                )
                .await?;
                apply_inlined_file_deletes(
                    &mut tx,
                    write.table_id,
                    snapshot_id,
                    write.base_snapshot_id,
                    &write.inlined_file_deletes,
                )
                .await?;
                drop_data_files(
                    &mut tx,
                    write.table_id,
                    snapshot_id,
                    write.base_snapshot_id,
                    &write.dropped_files,
                )
                .await?;
                let changes_made = staged_table_write_changes(write, had_files, had_inlined_rows);
                record_snapshot_changes(&mut tx, snapshot_id, &changes_made, commit_metadata)
                    .await?;
                let schema_id: i64 =
                    sqlx::query_scalar("SELECT schema_id FROM ducklake_table WHERE table_id = ?")
                        .bind(write.table_id)
                        .fetch_one(&mut *tx)
                        .await?;
                tables.push(CommitIds {
                    snapshot_id,
                    schema_id,
                    table_id: write.table_id,
                });
            }
            tx.commit().await?;
            Ok(MultiTableCommit {
                snapshot_id,
                tables,
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn set_delete_file(
        &self,
        table_id: i64,
        _schema_name: &str,
        _table_name: &str,
        _snapshot_id: i64,
        data_file_id: i64,
        expected_prev_delete_file: Option<i64>,
        base_snapshot: i64,
        delete: &DeleteFileInfo,
    ) -> Result<CommitIds> {
        block_on(async {
            // Single atomic commit: allocate the snapshot (write-lock-first),
            // fence against a concurrent generation change, compare-and-swap the
            // live delete file for this data file, retire the prior one, and
            // insert the new cumulative delete file — so at most one delete file
            // is ever live per data file and the head only resolves to a
            // fully-populated snapshot.
            let mut tx = self.pool.begin().await?;

            // First write takes the SQLite write lock up front (see
            // `insert_snapshot`); a delete carries `schema_version` forward.
            let (snapshot_id, _schema_version) = insert_snapshot(&mut tx).await?;
            // Official's table-level check: a drop, alter, compaction or insert
            // of the table since `base_snapshot` refuses the delete.
            let ours = TableChanges {
                deleted_from: true,
                ..TableChanges::default()
            };
            check_table_conflicts(&mut tx, table_id, base_snapshot, snapshot_id, ours).await?;

            // Target-file fence: the resolved positions are physical row indices
            // in `data_file_id`, and a parquet data file is immutable — so only a
            // concurrent write that RETIRED this file (a Replace/compaction) since
            // `base_snapshot` can invalidate them; a concurrent delete on THIS
            // file is caught by the compare-and-swap below. Abort iff the target
            // is no longer the live file.
            let target_live: Option<i64> = sqlx::query_scalar(
                "SELECT 1 FROM ducklake_data_file
                 WHERE data_file_id = ? AND end_snapshot IS NULL",
            )
            .bind(data_file_id)
            .fetch_optional(&mut *tx)
            .await?;
            if target_live.is_none() {
                return Err(crate::DuckLakeError::Conflict(format!(
                    "delete targets data file {data_file_id}, which was retired by a \
                     concurrent write since snapshot {base_snapshot}; retry against the \
                     new generation"
                )));
            }

            // Compare-and-swap on the currently-live delete file for this data
            // file (`end_snapshot IS NULL`). If it isn't what the caller saw, a
            // concurrent delete on the same data file won — abort.
            let current_prev: Option<i64> = sqlx::query_scalar(
                "SELECT delete_file_id FROM ducklake_delete_file
                 WHERE data_file_id = ? AND end_snapshot IS NULL",
            )
            .bind(data_file_id)
            .fetch_optional(&mut *tx)
            .await?;
            if current_prev != expected_prev_delete_file {
                return Err(crate::DuckLakeError::Conflict(format!(
                    "delete on data file {data_file_id} conflicts with a concurrent delete \
                     (expected live delete file {expected_prev_delete_file:?}, found \
                     {current_prev:?}); retry against the new generation"
                )));
            }

            // Retire the prior delete file (cumulative: the new file carries all
            // still-deleted positions, so the old one is superseded).
            if let Some(prev) = expected_prev_delete_file {
                sqlx::query(
                    "UPDATE ducklake_delete_file SET end_snapshot = ?
                     WHERE delete_file_id = ? AND end_snapshot IS NULL",
                )
                .bind(snapshot_id)
                .bind(prev)
                .execute(&mut *tx)
                .await?;
            }

            let file_id = next_file_id(&mut tx).await?;
            sqlx::query(
                "INSERT INTO ducklake_delete_file
                     (delete_file_id, data_file_id, table_id, path, path_is_relative, file_size_bytes,
                      footer_size, delete_count, begin_snapshot)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(file_id)
            .bind(data_file_id)
            .bind(table_id)
            .bind(&delete.path)
            .bind(delete.path_is_relative)
            .bind(delete.file_size_bytes)
            .bind(delete.footer_size)
            .bind(delete.delete_count)
            .bind(snapshot_id)
            .execute(&mut *tx)
            .await?;

            let changes_made = format!("deleted_from_table:{table_id}");
            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &changes_made,
                &SnapshotCommitMetadata::default(),
            )
            .await?;

            let schema_id: i64 =
                sqlx::query_scalar("SELECT schema_id FROM ducklake_table WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;

            tx.commit().await?;
            Ok(CommitIds {
                snapshot_id,
                schema_id,
                table_id,
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn register_data_file_with_deletes(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        snapshot_id: i64,
        file: &DataFileInfo,
        deletes: &[DeleteFileEntry],
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
    ) -> Result<CommitIds> {
        self.register_data_file_with_deletes_and_commit_metadata(
            table_id,
            schema_name,
            table_name,
            snapshot_id,
            file,
            deletes,
            mode,
            base_snapshot,
            columns,
            column_ids,
            &SnapshotCommitMetadata::default(),
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn register_data_file_with_deletes_and_commit_metadata(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        _snapshot_id: i64,
        file: &DataFileInfo,
        deletes: &[DeleteFileEntry],
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
        commit_metadata: &SnapshotCommitMetadata,
        expected_base_snapshot_id: Option<i64>,
    ) -> Result<CommitIds> {
        validate_delete_entries(mode, deletes)?;
        block_on(async {
            // One atomic commit for a combined append + positional deletes (an
            // update/upsert). finalize_snapshot allocates the snapshot + finalizes
            // the column generation; the new data file AND every delete file are
            // stamped with that one id and committed together, so the head only
            // ever resolves to the fully-applied mutation.
            let mut tx = self.pool.begin().await?;

            let snapshot_id = finalize_snapshot(
                &mut tx,
                table_id,
                schema_name,
                table_name,
                columns,
                column_ids,
                mode,
                base_snapshot,
            )
            .await?;
            if mode != WriteMode::Replace
                && let Some(expected_base_snapshot_id) = expected_base_snapshot_id
            {
                detect_replace_conflict(
                    &mut tx,
                    table_id,
                    expected_base_snapshot_id,
                    ConflictScope::Table,
                )
                .await?;
            }
            if !deletes.is_empty() {
                let ours = TableChanges {
                    inserted_into: true,
                    deleted_from: true,
                    ..TableChanges::default()
                };
                let source = deleted_files_table(&mut tx, table_id, deletes).await?;
                check_table_conflicts(&mut tx, source, base_snapshot, snapshot_id, ours).await?;
            }
            if !deletes.is_empty() {
                let sources = SourceFile::of_entries(deletes, false);
                detect_new_inlined_deletes(&mut tx, table_id, base_snapshot, &sources).await?;
            }

            // Partition-spec fence: the new row versions are a NEW write, so they
            // must agree with the table's live partition generation exactly as
            // register_data_file's do.
            let live_partition_id: Option<i64> = sqlx::query_scalar(
                "SELECT partition_id FROM ducklake_partition_info
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(table_id)
            .fetch_optional(&mut *tx)
            .await?;
            crate::metadata_writer::enforce_partition_fence(table_id, live_partition_id, file)?;

            // Register the new data file (inserted row versions), as in
            // register_data_file. Deletes are accounted at read time
            // (delete_count), so record_count stays gross — no adjustment for them.
            sqlx::query(
                "INSERT OR IGNORE INTO ducklake_table_stats
                     (table_id, record_count, next_row_id, file_size_bytes)
                 VALUES (?, 0, 0, 0)",
            )
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            let stats_row =
                sqlx::query("SELECT next_row_id FROM ducklake_table_stats WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;
            let next_row_id: i64 = stats_row.try_get(0)?;
            let row_ids = crate::metadata_writer::allocate_row_id_start(next_row_id, file);

            let file_id = next_file_id(&mut tx).await?;
            sqlx::query(
                "INSERT INTO ducklake_data_file
                     (data_file_id, table_id, path, path_is_relative, file_size_bytes,
                      footer_size, record_count, row_id_start, begin_snapshot)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(file_id)
            .bind(table_id)
            .bind(&file.path)
            .bind(file.path_is_relative)
            .bind(file.file_size_bytes)
            .bind(file.footer_size)
            .bind(file.record_count)
            .bind(row_ids.stored)
            .bind(snapshot_id)
            .execute(&mut *tx)
            .await?;

            // Persist the new file's per-column zone-map stats.
            let data_file_id = file_id;
            insert_file_column_stats(&mut tx, table_id, data_file_id, &file.column_stats).await?;
            insert_partition_metadata(&mut tx, table_id, data_file_id, file).await?;
            // Merge the file's statistics into the table-wide roll-up.
            merge_file_table_column_stats(
                &mut tx,
                table_id,
                &crate::metadata_writer::catalog_fields(columns, column_ids)?,
                &crate::metadata_writer::file_column_stats([file]),
            )
            .await?;

            sqlx::query(
                "UPDATE ducklake_table_stats
                 SET next_row_id     = next_row_id + ?,
                     record_count    = record_count + ?,
                     file_size_bytes = file_size_bytes + ?
                 WHERE table_id = ?",
            )
            .bind(row_ids.advance)
            .bind(file.record_count)
            .bind(file.file_size_bytes)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            // Apply each positional delete with the same fence + compare-and-swap as
            // set_delete_file, stamped with this snapshot. Each entry targets a
            // distinct data file, so there is no intra-transaction CAS contention.
            for entry in deletes {
                let target_live: Option<i64> = sqlx::query_scalar(
                    "SELECT 1 FROM ducklake_data_file
                     WHERE data_file_id = ? AND end_snapshot IS NULL",
                )
                .bind(entry.data_file_id)
                .fetch_optional(&mut *tx)
                .await?;
                if target_live.is_none() {
                    return Err(crate::DuckLakeError::Conflict(format!(
                        "UPDATE/DELETE on data file {} could not commit: the file is no longer \
                         live as of the catalog's current head (retired since snapshot \
                         {base_snapshot}). This happens when another writer committed a \
                         Replace/compaction. Retry the statement.",
                        entry.data_file_id
                    )));
                }

                let current_prev: Option<i64> = sqlx::query_scalar(
                    "SELECT delete_file_id FROM ducklake_delete_file
                     WHERE data_file_id = ? AND end_snapshot IS NULL",
                )
                .bind(entry.data_file_id)
                .fetch_optional(&mut *tx)
                .await?;
                if current_prev != entry.expected_prev_delete_file {
                    return Err(crate::DuckLakeError::Conflict(format!(
                        "UPDATE/DELETE on data file {} could not commit: its live delete file \
                         changed from {:?} to {current_prev:?} since snapshot {base_snapshot}. \
                         Another writer committed a delete on this file. Retry the statement.",
                        entry.data_file_id, entry.expected_prev_delete_file
                    )));
                }

                if let Some(prev) = entry.expected_prev_delete_file {
                    sqlx::query(
                        "UPDATE ducklake_delete_file SET end_snapshot = ?
                         WHERE delete_file_id = ? AND end_snapshot IS NULL",
                    )
                    .bind(snapshot_id)
                    .bind(prev)
                    .execute(&mut *tx)
                    .await?;
                }

                let file_id = next_file_id(&mut tx).await?;
                sqlx::query(
                    "INSERT INTO ducklake_delete_file
                         (delete_file_id, data_file_id, table_id, path, path_is_relative, file_size_bytes,
                          footer_size, delete_count, begin_snapshot)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                )
                .bind(file_id)
                .bind(entry.data_file_id)
                .bind(table_id)
                .bind(&entry.delete.path)
                .bind(entry.delete.path_is_relative)
                .bind(entry.delete.file_size_bytes)
                .bind(entry.delete.footer_size)
                .bind(entry.delete.delete_count)
                .bind(snapshot_id)
                .execute(&mut *tx)
                .await?;
            }

            let replaced_existing_data = replaced_storage(&mut tx, table_id, snapshot_id).await?;
            let changes_made =
                table_write_changes(table_id, mode, !deletes.is_empty(), replaced_existing_data);
            record_snapshot_changes(&mut tx, snapshot_id, &changes_made, commit_metadata).await?;

            let schema_id: i64 =
                sqlx::query_scalar("SELECT schema_id FROM ducklake_table WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;

            tx.commit().await?;
            Ok(CommitIds {
                snapshot_id,
                schema_id,
                table_id,
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn register_data_files_with_deletes(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        snapshot_id: i64,
        files: &[DataFileInfo],
        deletes: &[DeleteFileEntry],
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
    ) -> Result<CommitIds> {
        self.register_data_files_with_deletes_and_commit_metadata(
            table_id,
            schema_name,
            table_name,
            snapshot_id,
            files,
            deletes,
            mode,
            base_snapshot,
            columns,
            column_ids,
            &SnapshotCommitMetadata::default(),
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn register_data_files_with_deletes_and_commit_metadata(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        _snapshot_id: i64,
        files: &[DataFileInfo],
        deletes: &[DeleteFileEntry],
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
        commit_metadata: &SnapshotCommitMetadata,
        expected_base_snapshot_id: Option<i64>,
    ) -> Result<CommitIds> {
        if files.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "register_data_files_with_deletes: files must be non-empty".to_string(),
            ));
        }
        validate_delete_entries(mode, deletes)?;
        block_on(async {
            // One atomic commit for a combined N-file append + positional deletes
            // (a keyed mutation whose new row versions span several partitions or
            // rolled files). finalize_snapshot allocates the snapshot + finalizes
            // the column generation; EVERY appended data file AND every delete file
            // are stamped with that one id and committed together, so the head only
            // ever resolves to the fully-applied mutation.
            let mut tx = self.pool.begin().await?;

            let snapshot_id = finalize_snapshot(
                &mut tx,
                table_id,
                schema_name,
                table_name,
                columns,
                column_ids,
                mode,
                base_snapshot,
            )
            .await?;
            if mode != WriteMode::Replace
                && let Some(expected_base_snapshot_id) = expected_base_snapshot_id
            {
                detect_replace_conflict(
                    &mut tx,
                    table_id,
                    expected_base_snapshot_id,
                    ConflictScope::Table,
                )
                .await?;
            }
            if !deletes.is_empty() {
                let ours = TableChanges {
                    inserted_into: !files.is_empty(),
                    deleted_from: true,
                    ..TableChanges::default()
                };
                let source = deleted_files_table(&mut tx, table_id, deletes).await?;
                check_table_conflicts(&mut tx, source, base_snapshot, snapshot_id, ours).await?;
            }
            if !deletes.is_empty() {
                let sources = SourceFile::of_entries(deletes, false);
                detect_new_inlined_deletes(&mut tx, table_id, base_snapshot, &sources).await?;
            }

            // Partition-spec fence (both directions, every file): the new row
            // versions are a NEW write, so each must agree with the table's live
            // partition generation exactly as register_data_files' do.
            let live_partition_id: Option<i64> = sqlx::query_scalar(
                "SELECT partition_id FROM ducklake_partition_info
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(table_id)
            .fetch_optional(&mut *tx)
            .await?;
            for file in files {
                crate::metadata_writer::enforce_partition_fence(table_id, live_partition_id, file)?;
            }

            // Register the new data files (inserted row versions), as in
            // register_data_files. Deletes are accounted at read time
            // (delete_count), so record_count stays gross — no adjustment for them.
            sqlx::query(
                "INSERT OR IGNORE INTO ducklake_table_stats
                     (table_id, record_count, next_row_id, file_size_bytes)
                 VALUES (?, 0, 0, 0)",
            )
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            let mut next_row_id: i64 =
                sqlx::query("SELECT next_row_id FROM ducklake_table_stats WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?
                    .try_get(0)?;
            let mut total_advance: i64 = 0;
            let mut total_records: i64 = 0;
            let mut total_bytes: i64 = 0;
            for file in files {
                let row_ids = crate::metadata_writer::allocate_row_id_start(next_row_id, file);
                let file_id = next_file_id(&mut tx).await?;
                sqlx::query(
                    "INSERT INTO ducklake_data_file
                         (data_file_id, table_id, path, path_is_relative, file_size_bytes,
                          footer_size, record_count, row_id_start, begin_snapshot)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                )
                .bind(file_id)
                .bind(table_id)
                .bind(&file.path)
                .bind(file.path_is_relative)
                .bind(file.file_size_bytes)
                .bind(file.footer_size)
                .bind(file.record_count)
                .bind(row_ids.stored)
                .bind(snapshot_id)
                .execute(&mut *tx)
                .await?;
                let data_file_id = file_id;
                insert_file_column_stats(&mut tx, table_id, data_file_id, &file.column_stats)
                    .await?;
                insert_partition_metadata(&mut tx, table_id, data_file_id, file).await?;
                next_row_id += row_ids.advance;
                total_advance += row_ids.advance;
                total_records += file.record_count;
                total_bytes += file.file_size_bytes;
            }
            // Merge the files' statistics into the table-wide roll-up.
            merge_file_table_column_stats(
                &mut tx,
                table_id,
                &crate::metadata_writer::catalog_fields(columns, column_ids)?,
                &crate::metadata_writer::file_column_stats(files),
            )
            .await?;

            sqlx::query(
                "UPDATE ducklake_table_stats
                 SET next_row_id     = next_row_id + ?,
                     record_count    = record_count + ?,
                     file_size_bytes = file_size_bytes + ?
                 WHERE table_id = ?",
            )
            .bind(total_advance)
            .bind(total_records)
            .bind(total_bytes)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            // Apply each positional delete with the same fence + compare-and-swap as
            // set_delete_file, stamped with this snapshot. Each entry targets a
            // distinct data file, so there is no intra-transaction CAS contention.
            for entry in deletes {
                let target_live: Option<i64> = sqlx::query_scalar(
                    "SELECT 1 FROM ducklake_data_file
                     WHERE data_file_id = ? AND end_snapshot IS NULL",
                )
                .bind(entry.data_file_id)
                .fetch_optional(&mut *tx)
                .await?;
                if target_live.is_none() {
                    return Err(crate::DuckLakeError::Conflict(format!(
                        "UPDATE/DELETE on data file {} could not commit: the file is no longer \
                         live as of the catalog's current head (retired since snapshot \
                         {base_snapshot}). This happens when another writer committed a \
                         Replace/compaction. Retry the statement.",
                        entry.data_file_id
                    )));
                }

                let current_prev: Option<i64> = sqlx::query_scalar(
                    "SELECT delete_file_id FROM ducklake_delete_file
                     WHERE data_file_id = ? AND end_snapshot IS NULL",
                )
                .bind(entry.data_file_id)
                .fetch_optional(&mut *tx)
                .await?;
                if current_prev != entry.expected_prev_delete_file {
                    return Err(crate::DuckLakeError::Conflict(format!(
                        "UPDATE/DELETE on data file {} could not commit: its live delete file \
                         changed from {:?} to {current_prev:?} since snapshot {base_snapshot}. \
                         Another writer committed a delete on this file. Retry the statement.",
                        entry.data_file_id, entry.expected_prev_delete_file
                    )));
                }

                if let Some(prev) = entry.expected_prev_delete_file {
                    sqlx::query(
                        "UPDATE ducklake_delete_file SET end_snapshot = ?
                         WHERE delete_file_id = ? AND end_snapshot IS NULL",
                    )
                    .bind(snapshot_id)
                    .bind(prev)
                    .execute(&mut *tx)
                    .await?;
                }

                let file_id = next_file_id(&mut tx).await?;
                sqlx::query(
                    "INSERT INTO ducklake_delete_file
                         (delete_file_id, data_file_id, table_id, path, path_is_relative, file_size_bytes,
                          footer_size, delete_count, begin_snapshot)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                )
                .bind(file_id)
                .bind(entry.data_file_id)
                .bind(table_id)
                .bind(&entry.delete.path)
                .bind(entry.delete.path_is_relative)
                .bind(entry.delete.file_size_bytes)
                .bind(entry.delete.footer_size)
                .bind(entry.delete.delete_count)
                .bind(snapshot_id)
                .execute(&mut *tx)
                .await?;
            }

            let replaced_existing_data = replaced_storage(&mut tx, table_id, snapshot_id).await?;
            let changes_made =
                table_write_changes(table_id, mode, !deletes.is_empty(), replaced_existing_data);
            record_snapshot_changes(&mut tx, snapshot_id, &changes_made, commit_metadata).await?;

            let schema_id: i64 =
                sqlx::query_scalar("SELECT schema_id FROM ducklake_table WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;

            tx.commit().await?;
            Ok(CommitIds {
                snapshot_id,
                schema_id,
                table_id,
            })
        })
    }

    fn commit_positional_deletes(
        &self,
        table_id: i64,
        _schema_name: &str,
        _table_name: &str,
        base_snapshot: i64,
        deletes: &[DeleteFileEntry],
    ) -> Result<CommitIds> {
        if deletes.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "commit_positional_deletes requires at least one delete entry".to_string(),
            ));
        }
        // A positional delete never retires the data files it targets, so it is
        // Append-semantics for validation (also enforces distinct data files).
        validate_delete_entries(WriteMode::Append, deletes)?;
        block_on(async {
            // One atomic commit for an N-file positional DELETE with no append.
            // Mirrors register_data_file_with_deletes' delete loop, but allocates
            // the snapshot via insert_snapshot (no column generation, no data
            // file) — a delete carries schema_version forward. The head
            // (MAX(snapshot_id)) only ever resolves to the fully-applied delete.
            let mut tx = self.pool.begin().await?;

            // Write-lock-first: the MAX+1 insert takes the SQLite write lock up
            // front, so concurrent writers can't collide or deadlock.
            let (snapshot_id, _schema_version) = insert_snapshot(&mut tx).await?;
            let ours = TableChanges {
                deleted_from: true,
                ..TableChanges::default()
            };
            check_table_conflicts(&mut tx, table_id, base_snapshot, snapshot_id, ours).await?;
            let sources = SourceFile::of_entries(deletes, true);
            detect_new_inlined_deletes(&mut tx, table_id, base_snapshot, &sources).await?;

            for entry in deletes {
                // Target-file fence: abort iff the data file is no longer live (a
                // concurrent Replace/compaction retired it, invalidating the
                // resolved positions).
                let target_live: Option<i64> = sqlx::query_scalar(
                    "SELECT 1 FROM ducklake_data_file
                     WHERE data_file_id = ? AND end_snapshot IS NULL",
                )
                .bind(entry.data_file_id)
                .fetch_optional(&mut *tx)
                .await?;
                if target_live.is_none() {
                    return Err(crate::DuckLakeError::Conflict(format!(
                        "DELETE on data file {} could not commit: the file is no longer live as \
                         of the catalog's current head (retired since snapshot {base_snapshot}). \
                         This happens when another writer committed a Replace/compaction. Retry the statement.",
                        entry.data_file_id
                    )));
                }

                // Compare-and-swap on the currently-live delete file for this data
                // file; a concurrent delete on the same file makes it differ.
                let current_prev: Option<i64> = sqlx::query_scalar(
                    "SELECT delete_file_id FROM ducklake_delete_file
                     WHERE data_file_id = ? AND end_snapshot IS NULL",
                )
                .bind(entry.data_file_id)
                .fetch_optional(&mut *tx)
                .await?;
                if current_prev != entry.expected_prev_delete_file {
                    return Err(crate::DuckLakeError::Conflict(format!(
                        "DELETE on data file {} could not commit: its live delete file changed \
                         from {:?} to {current_prev:?} since snapshot {base_snapshot}. Another \
                         writer committed a delete on this file. Retry the statement.",
                        entry.data_file_id, entry.expected_prev_delete_file
                    )));
                }

                // Retire the prior delete file (cumulative: the new file carries
                // all still-deleted positions, so the old one is superseded).
                if let Some(prev) = entry.expected_prev_delete_file {
                    sqlx::query(
                        "UPDATE ducklake_delete_file SET end_snapshot = ?
                         WHERE delete_file_id = ? AND end_snapshot IS NULL",
                    )
                    .bind(snapshot_id)
                    .bind(prev)
                    .execute(&mut *tx)
                    .await?;
                }

                let file_id = next_file_id(&mut tx).await?;
                sqlx::query(
                    "INSERT INTO ducklake_delete_file
                         (delete_file_id, data_file_id, table_id, path, path_is_relative, file_size_bytes,
                          footer_size, delete_count, begin_snapshot)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                )
                .bind(file_id)
                .bind(entry.data_file_id)
                .bind(table_id)
                .bind(&entry.delete.path)
                .bind(entry.delete.path_is_relative)
                .bind(entry.delete.file_size_bytes)
                .bind(entry.delete.footer_size)
                .bind(entry.delete.delete_count)
                .bind(snapshot_id)
                .execute(&mut *tx)
                .await?;
            }

            let changes_made = format!("deleted_from_table:{table_id}");
            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &changes_made,
                &SnapshotCommitMetadata::default(),
            )
            .await?;

            let schema_id: i64 =
                sqlx::query_scalar("SELECT schema_id FROM ducklake_table WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;

            tx.commit().await?;
            Ok(CommitIds {
                snapshot_id,
                schema_id,
                table_id,
            })
        })
    }

    fn commit_inlined_deletes(
        &self,
        table_id: i64,
        _schema_name: &str,
        _table_name: &str,
        base_snapshot: i64,
        rows: &[InlinedRowRef],
    ) -> Result<CommitIds> {
        if rows.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "commit_inlined_deletes requires at least one row".to_string(),
            ));
        }
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let (snapshot_id, _schema_version) = insert_snapshot(&mut tx).await?;
            let ours = TableChanges {
                deleted_inlined: true,
                ..TableChanges::default()
            };
            check_table_conflicts(&mut tx, table_id, base_snapshot, snapshot_id, ours).await?;
            apply_inlined_deletes(&mut tx, table_id, snapshot_id, base_snapshot, rows).await?;
            let changes_made = format!("inlined_delete:{table_id}");
            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &changes_made,
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            let schema_id: i64 =
                sqlx::query_scalar("SELECT schema_id FROM ducklake_table WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;
            tx.commit().await?;
            Ok(CommitIds {
                snapshot_id,
                schema_id,
                table_id,
            })
        })
    }

    fn commit_deletes(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        base_snapshot: i64,
        positional: &[DeleteFileEntry],
        inlined: &[InlinedRowRef],
    ) -> Result<CommitIds> {
        if inlined.is_empty() {
            return self.commit_positional_deletes(
                table_id,
                schema_name,
                table_name,
                base_snapshot,
                positional,
            );
        }
        if positional.is_empty() {
            return self.commit_inlined_deletes(
                table_id,
                schema_name,
                table_name,
                base_snapshot,
                inlined,
            );
        }
        validate_delete_entries(WriteMode::Append, positional)?;
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let (snapshot_id, _schema_version) = insert_snapshot(&mut tx).await?;
            let ours = TableChanges {
                deleted_from: !positional.is_empty(),
                deleted_inlined: !inlined.is_empty(),
                ..TableChanges::default()
            };
            check_table_conflicts(&mut tx, table_id, base_snapshot, snapshot_id, ours).await?;
            apply_positional_deletes(&mut tx, table_id, snapshot_id, base_snapshot, positional)
                .await?;
            apply_inlined_deletes(&mut tx, table_id, snapshot_id, base_snapshot, inlined).await?;
            let changes_made =
                table_storage_changes(table_id, None, !positional.is_empty(), !inlined.is_empty());
            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &changes_made,
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            let schema_id: i64 =
                sqlx::query_scalar("SELECT schema_id FROM ducklake_table WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;
            tx.commit().await?;
            Ok(CommitIds {
                snapshot_id,
                schema_id,
                table_id,
            })
        })
    }

    fn commit_compaction(
        &self,
        table_id: i64,
        base_snapshot: i64,
        sources: &[crate::metadata_writer::CompactionSourceFile],
        outputs: &[crate::metadata_writer::CompactionOutputFile],
        retirement: crate::metadata_writer::SourceRetirement,
    ) -> Result<CommitIds> {
        use crate::metadata_writer::SourceRetirement;
        if sources.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "commit_compaction requires at least one source file".to_string(),
            ));
        }
        block_on(async {
            // One atomic snapshot for the whole compaction. insert_snapshot takes
            // the SQLite write lock up front (write-lock-first) and carries
            // schema_version forward — compaction is not DDL. The head
            // (MAX(snapshot_id)) only ever resolves to the fully-applied layout.
            let mut tx = self.pool.begin().await?;
            let (snapshot_id, _schema_version) = insert_snapshot(&mut tx).await?;
            let source_data_ids = sources
                .iter()
                .map(|source| source.data_file_id)
                .collect::<Vec<_>>();
            let source_files = source_data_ids
                .iter()
                .map(|id| SourceFile::any(*id))
                .collect::<Vec<_>>();
            detect_new_inlined_deletes(&mut tx, table_id, base_snapshot, &source_files).await?;

            // Conflict fence per source: the data file must still be live, and its
            // live delete file must still match what the caller read the source's
            // rows against. A concurrent APPEND adds unrelated files and trips
            // neither (so compaction coexists with it); a concurrent
            // Replace/compaction that retired the file, or a DELETE/UPDATE that
            // changed its live rows, DOES — abort before mutating anything so a
            // retired/deleted row can never be resurrected into an output.
            for src in sources {
                let target_live: Option<i64> = sqlx::query_scalar(
                    "SELECT 1 FROM ducklake_data_file
                     WHERE data_file_id = ? AND table_id = ? AND end_snapshot IS NULL",
                )
                .bind(src.data_file_id)
                .bind(table_id)
                .fetch_optional(&mut *tx)
                .await?;
                if target_live.is_none() {
                    return Err(crate::DuckLakeError::Conflict(format!(
                        "compaction of table {table_id} could not commit: source data file {} is \
                         no longer live (retired by a concurrent Replace/compaction since \
                         snapshot {base_snapshot}). Re-open the catalog at the latest snapshot \
                         and re-plan.",
                        src.data_file_id
                    )));
                }

                let current_delete: Option<i64> = sqlx::query_scalar(
                    "SELECT delete_file_id FROM ducklake_delete_file
                     WHERE data_file_id = ? AND end_snapshot IS NULL",
                )
                .bind(src.data_file_id)
                .fetch_optional(&mut *tx)
                .await?;
                if current_delete != src.delete_file_id {
                    return Err(crate::DuckLakeError::Conflict(format!(
                        "compaction of table {table_id} could not commit: the live delete file of \
                         source data file {} changed from {:?} to {current_delete:?} since \
                         snapshot {base_snapshot} (a concurrent DELETE/UPDATE). Re-open the \
                         catalog at the latest snapshot and re-plan.",
                        src.data_file_id, src.delete_file_id
                    )));
                }
            }

            match retirement {
                SourceRetirement::Remove => {
                    // Merge: the partial output serves every snapshot the sources
                    // did, so the sources are redundant. Schedule their physical
                    // files for deletion (resolving paths as expire_snapshots
                    // does) and REMOVE their catalog rows, so no snapshot resolves
                    // to them (avoids double-counting with the partial file, and
                    // upholds the invariant that scheduled files are unreachable).
                    let dead_data = sqlx::query(AssertSqlSafe(format!(
                        "SELECT df.data_file_id, {RESOLVED_PATH} AS resolved_path, {REL_FLAG} AS rel
                         FROM ducklake_data_file df
                         JOIN ducklake_table t ON t.table_id = df.table_id
                         JOIN ducklake_schema s ON s.schema_id = t.schema_id
                         WHERE df.data_file_id IN ({})",
                        id_list(&source_data_ids)
                    )))
                    .fetch_all(&mut *tx)
                    .await?;
                    schedule_files(&mut tx, dead_data).await?;

                    let dead_del = sqlx::query(AssertSqlSafe(format!(
                        "SELECT df.delete_file_id, {RESOLVED_PATH} AS resolved_path, {REL_FLAG} AS rel
                         FROM ducklake_delete_file df
                         JOIN ducklake_table t ON t.table_id = df.table_id
                         JOIN ducklake_schema s ON s.schema_id = t.schema_id
                         WHERE df.data_file_id IN ({})",
                        id_list(&source_data_ids)
                    )))
                    .fetch_all(&mut *tx)
                    .await?;
                    schedule_files(&mut tx, dead_del).await?;

                    sqlx::query(AssertSqlSafe(format!(
                        "DELETE FROM ducklake_delete_file WHERE data_file_id IN ({})",
                        id_list(&source_data_ids)
                    )))
                    .execute(&mut *tx)
                    .await?;
                    sqlx::query(AssertSqlSafe(format!(
                        "DELETE FROM ducklake_data_file WHERE data_file_id IN ({})",
                        id_list(&source_data_ids)
                    )))
                    .execute(&mut *tx)
                    .await?;
                    // Hard-delete the removed sources' per-column stats too, as
                    // official DuckLake does on merge (otherwise they orphan).
                    sqlx::query(AssertSqlSafe(format!(
                        "DELETE FROM ducklake_file_column_stats WHERE data_file_id IN ({})",
                        id_list(&source_data_ids)
                    )))
                    .execute(&mut *tx)
                    .await?;
                    // Likewise the removed sources' per-file partition values.
                    sqlx::query(AssertSqlSafe(format!(
                        "DELETE FROM ducklake_file_partition_value WHERE data_file_id IN ({})",
                        id_list(&source_data_ids)
                    )))
                    .execute(&mut *tx)
                    .await?;
                },
                SourceRetirement::Retire => {
                    // Rewrite: the output only holds currently-live rows, so the
                    // sources still serve time travel to pre-compaction snapshots.
                    // Retire them (end_snapshot) but do NOT schedule them — an
                    // expire_snapshots run schedules them once their snapshots are
                    // gone, so they are never deleted while still reachable.
                    sqlx::query(AssertSqlSafe(format!(
                        "UPDATE ducklake_data_file SET end_snapshot = ?
                         WHERE data_file_id IN ({}) AND end_snapshot IS NULL",
                        id_list(&source_data_ids)
                    )))
                    .bind(snapshot_id)
                    .execute(&mut *tx)
                    .await?;
                    sqlx::query(AssertSqlSafe(format!(
                        "UPDATE ducklake_delete_file SET end_snapshot = ?
                         WHERE data_file_id IN ({}) AND end_snapshot IS NULL",
                        id_list(&source_data_ids)
                    )))
                    .bind(snapshot_id)
                    .execute(&mut *tx)
                    .await?;
                },
            }

            // Register each rewritten output. begin_snapshot = the file's min
            // origin snapshot for a merged partial file (so historical reads see
            // it, row-filtered by origin), else this compaction snapshot;
            // row_id_start = the carried range of a rowid-adjacent merge, else
            // NULL (rowids are served from the embedded rowid column);
            // partial_max marks a merged partial file.
            for out in outputs {
                let begin = out.begin_snapshot.unwrap_or(snapshot_id);
                let file_id = next_file_id(&mut tx).await?;
                sqlx::query(
                    "INSERT INTO ducklake_data_file
                         (data_file_id, table_id, path, path_is_relative, file_size_bytes,
                          footer_size, record_count, row_id_start, begin_snapshot, partial_max)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                )
                .bind(file_id)
                .bind(table_id)
                .bind(&out.file.path)
                .bind(out.file.path_is_relative)
                .bind(out.file.file_size_bytes)
                .bind(out.file.footer_size)
                .bind(out.file.record_count)
                .bind(crate::metadata_writer::carried_row_id_start(&out.file))
                .bind(begin)
                .bind(out.partial_max)
                .execute(&mut *tx)
                .await?;
                // Persist the compacted file's harvested zone maps under its id
                // (same tx). Mirrors how official DuckLake records stats for
                // compaction outputs via the shared write path.
                let data_file_id = file_id;
                insert_file_column_stats(&mut tx, table_id, data_file_id, &out.file.column_stats)
                    .await?;
                insert_partition_metadata(&mut tx, table_id, data_file_id, &out.file).await?;
            }

            // `ducklake_table_stats` is left as it is, as official's compaction
            // leaves it: `record_count` stays the gross count of rows ever
            // inserted (DuckDB answers MIN/MAX from the widen-only column bounds
            // only while it equals the live count), and next_row_id does not move
            // because the outputs re-embed existing rowids.

            // Attribute the snapshot for spec readers (DuckDB reads this ledger).
            let changes_made = crate::metadata_writer::compaction_change(retirement, table_id);
            let commit_metadata = SnapshotCommitMetadata::new();
            record_snapshot_changes(&mut tx, snapshot_id, &changes_made, &commit_metadata).await?;

            let schema_id: i64 =
                sqlx::query_scalar("SELECT schema_id FROM ducklake_table WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;

            tx.commit().await?;
            Ok(CommitIds {
                snapshot_id,
                schema_id,
                table_id,
            })
        })
    }

    fn commit_inlined_flush(
        &self,
        base_snapshot: i64,
        tables: &[InlinedFlushTable],
    ) -> Result<i64> {
        validate_inlined_flush(tables)?;
        block_on(async {
            let mut tx = self.pool.begin().await?;
            // Write-lock-first; a flush is not DDL, so schema_version carries forward.
            let (snapshot_id, _schema_version) = insert_snapshot(&mut tx).await?;
            let changes: Vec<Option<String>> = sqlx::query_scalar(
                "SELECT changes_made FROM ducklake_snapshot_changes
                 WHERE snapshot_id > ? AND snapshot_id < ?",
            )
            .bind(base_snapshot)
            .bind(snapshot_id)
            .fetch_all(&mut *tx)
            .await?;
            if let Some(conflict) =
                crate::metadata_writer::inlined_flush_table_conflict(tables, &changes)
            {
                return Err(conflict);
            }
            for table in tables {
                let live: Option<i64> = sqlx::query_scalar(
                    "SELECT 1 FROM ducklake_table WHERE table_id = ? AND end_snapshot IS NULL",
                )
                .bind(table.table_id)
                .fetch_optional(&mut *tx)
                .await?;
                if live.is_none() {
                    return Err(inlined_flush_conflict(table.table_id, base_snapshot));
                }
                sqlx::query(
                    "INSERT OR IGNORE INTO ducklake_table_stats
                         (table_id, record_count, next_row_id, file_size_bytes)
                     VALUES (?, 0, 0, 0)",
                )
                .bind(table.table_id)
                .execute(&mut *tx)
                .await?;
                let mut added_bytes = 0;
                for out in &table.files {
                    let file_id = next_file_id(&mut tx).await?;
                    sqlx::query(
                        "INSERT INTO ducklake_data_file
                             (data_file_id, table_id, path, path_is_relative, file_size_bytes, footer_size,
                              record_count, row_id_start, begin_snapshot, partial_max)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    )
                    .bind(file_id)
                    .bind(table.table_id)
                    .bind(&out.file.path)
                    .bind(out.file.path_is_relative)
                    .bind(out.file.file_size_bytes)
                    .bind(out.file.footer_size)
                    .bind(out.file.record_count)
                    .bind(carried_row_id_start(&out.file))
                    .bind(out.begin_snapshot)
                    .bind(out.partial_max)
                    .execute(&mut *tx)
                    .await?;
                    let data_file_id = file_id;
                    insert_file_column_stats(
                        &mut tx,
                        table.table_id,
                        data_file_id,
                        &out.file.column_stats,
                    )
                    .await?;
                    insert_partition_metadata(&mut tx, table.table_id, data_file_id, &out.file)
                        .await?;
                    if let Some(delete) = &out.delete {
                        insert_flush_delete_file(
                            &mut tx,
                            table.table_id,
                            None,
                            data_file_id,
                            delete,
                        )
                        .await?;
                    }
                    added_bytes += out.file.file_size_bytes;
                }
                replace_delete_files(&mut tx, table.table_id, &table.file_deletes, || {
                    inlined_flush_conflict(table.table_id, base_snapshot)
                })
                .await?;
                if !table.file_deletes.is_empty() {
                    let inlined_delete =
                        crate::metadata_provider::inlined_delete_table_name(table.table_id)?;
                    let ids = id_list(
                        &table
                            .file_deletes
                            .iter()
                            .map(|entry| entry.data_file_id)
                            .collect::<Vec<_>>(),
                    );
                    sqlx::query(AssertSqlSafe(format!(
                        "DELETE FROM {} WHERE file_id IN ({ids}) AND begin_snapshot <= ?",
                        quote_ident(&inlined_delete)
                    )))
                    .bind(base_snapshot)
                    .execute(&mut *tx)
                    .await?;
                }
                for name in &table.inlined_tables {
                    let system = inlined_table_layout(&mut *tx, name).await?.system;
                    sqlx::query(AssertSqlSafe(format!(
                        "DELETE FROM {} WHERE {} <= ?",
                        quote_ident(name),
                        system.begin_snapshot,
                    )))
                    .bind(base_snapshot)
                    .execute(&mut *tx)
                    .await?;
                }
                if !table.files.is_empty() {
                    sqlx::query(
                        "UPDATE ducklake_table_stats SET file_size_bytes = file_size_bytes + ?
                         WHERE table_id = ?",
                    )
                    .bind(added_bytes)
                    .bind(table.table_id)
                    .execute(&mut *tx)
                    .await?;
                    let fields = live_stat_fields(table.table_id, &mut tx).await?;
                    merge_file_table_column_stats(
                        &mut tx,
                        table.table_id,
                        &fields,
                        &crate::metadata_writer::file_column_stats(
                            table.files.iter().map(|flushed| &flushed.file),
                        ),
                    )
                    .await?;
                }
            }
            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &inlined_flush_changes(tables),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            tx.commit().await?;
            Ok(snapshot_id)
        })
    }

    fn retire_appends_since(&self, table_id: i64, base_snapshot: i64) -> Result<Option<i64>> {
        block_on(async {
            // Metadata-only rollback of a pure-append delta in one snapshot: end
            // every live data file added after base_snapshot, recompute the visible
            // stats from the survivors. next_row_id is preserved (rowids never
            // reused). SQLite is single-catalog — head is implicit MAX(snapshot_id),
            // so there is no lock_catalog / advance_catalog_head (mirrors truncate).
            let mut tx = self.pool.begin().await?;

            // Write-lock-first (mirrors `commit_truncate`): allocate the snapshot up
            // front so the purity proof and no-op check run under SQLite's write lock
            // even if the caller hasn't externally serialized writes. On a Conflict or
            // no-op we return WITHOUT committing, so the allocation rolls back — no
            // snapshot row and no id gap. Non-DDL, so schema_version carries forward.
            let (snapshot_id, _schema_version) = insert_snapshot(&mut tx).await?;

            // Purity guard — see the trait doc. A post-base delete file, a base-era
            // data file ended after base, or a post-base column version means a
            // non-append mutation we cannot faithfully revert with forward-only
            // retirement → Conflict.
            // A delete file rewritten after base begins at its earliest deletion,
            // which may precede base; its `partial_max` is its latest one.
            let impure_delete: Option<i64> = sqlx::query_scalar(
                "SELECT 1 FROM ducklake_delete_file
                 WHERE table_id = ? AND (begin_snapshot > ? OR partial_max > ?) LIMIT 1",
            )
            .bind(table_id)
            .bind(base_snapshot)
            .bind(base_snapshot)
            .fetch_optional(&mut *tx)
            .await?;
            // A base-era file ended since, or a live file holding rows of both
            // sides of base (a flush's file begins at its earliest row).
            let impure_ended: Option<i64> = sqlx::query_scalar(
                "SELECT 1 FROM ducklake_data_file
                 WHERE table_id = ? AND begin_snapshot <= ?
                   AND ((end_snapshot IS NOT NULL AND end_snapshot > ?)
                        OR (end_snapshot IS NULL AND partial_max > ?)) LIMIT 1",
            )
            .bind(table_id)
            .bind(base_snapshot)
            .bind(base_snapshot)
            .bind(base_snapshot)
            .fetch_optional(&mut *tx)
            .await?;
            let impure_column: Option<i64> = sqlx::query_scalar(
                "SELECT 1 FROM ducklake_column
                 WHERE table_id = ? AND (begin_snapshot > ? OR end_snapshot > ?) LIMIT 1",
            )
            .bind(table_id)
            .bind(base_snapshot)
            .bind(base_snapshot)
            .fetch_optional(&mut *tx)
            .await?;
            // A partition-spec change (SET/RESET PARTITIONED BY) after base is DDL,
            // not a pure append, and forward-only file retirement cannot revert it.
            // (A pure append's per-file `ducklake_file_partition_value` rows are
            // fine — they belong to the appended files we retire, and are ignored
            // on read once those files are gone, exactly like file_column_stats.)
            let impure_partition: Option<i64> = sqlx::query_scalar(
                "SELECT 1 FROM ducklake_partition_info
                 WHERE table_id = ? AND (begin_snapshot > ? OR end_snapshot > ?) LIMIT 1",
            )
            .bind(table_id)
            .bind(base_snapshot)
            .bind(base_snapshot)
            .fetch_optional(&mut *tx)
            .await?;
            // Inlined rows: a row ended after base is a deletion (or an update's
            // old version), as is any inlined deletion of a Parquet row.
            let has_registry: Option<i64> = sqlx::query_scalar(
                "SELECT 1 FROM sqlite_master
                 WHERE type = 'table' AND name = 'ducklake_inlined_data_tables'",
            )
            .fetch_optional(&mut *tx)
            .await?;
            let inlined = if has_registry.is_some() {
                inlined_tables(&mut tx, table_id).await?
            } else {
                Vec::new()
            };
            let mut impure_inlined = false;
            for (name, system) in &inlined {
                let ended: Option<i64> = sqlx::query_scalar(AssertSqlSafe(format!(
                    "SELECT 1 FROM {} WHERE {} > ? LIMIT 1",
                    quote_ident(name),
                    system.end_snapshot,
                )))
                .bind(base_snapshot)
                .fetch_optional(&mut *tx)
                .await?;
                impure_inlined |= ended.is_some();
            }
            let inlined_deletes = crate::metadata_provider::inlined_delete_table_name(table_id)?;
            let has_inlined_deletes: Option<i64> =
                sqlx::query_scalar("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?")
                    .bind(&inlined_deletes)
                    .fetch_optional(&mut *tx)
                    .await?;
            if has_inlined_deletes.is_some() {
                let deleted: Option<i64> = sqlx::query_scalar(AssertSqlSafe(format!(
                    "SELECT 1 FROM {} WHERE begin_snapshot > ? LIMIT 1",
                    quote_ident(&inlined_deletes),
                )))
                .bind(base_snapshot)
                .fetch_optional(&mut *tx)
                .await?;
                impure_inlined |= deleted.is_some();
            }
            if impure_delete.is_some()
                || impure_ended.is_some()
                || impure_column.is_some()
                || impure_partition.is_some()
                || impure_inlined
            {
                return Err(crate::DuckLakeError::Conflict(format!(
                    "table {table_id}: changes since snapshot {base_snapshot} are not a pure \
                     append (delete/replace/update or schema/partition change present); refusing to retire"
                )));
            }

            // No-op → None (no snapshot) if nothing was appended after base.
            let appended_rows: Option<i64> = sqlx::query_scalar(
                "SELECT SUM(record_count) FROM ducklake_data_file
                 WHERE table_id = ? AND end_snapshot IS NULL AND begin_snapshot > ?",
            )
            .bind(table_id)
            .bind(base_snapshot)
            .fetch_one(&mut *tx)
            .await?;
            let mut appended_inlined = 0_i64;
            for (name, system) in &inlined {
                appended_inlined += sqlx::query_scalar::<_, i64>(AssertSqlSafe(format!(
                    "SELECT COUNT(*) FROM {} WHERE {} > ? AND {} IS NULL",
                    quote_ident(name),
                    system.begin_snapshot,
                    system.end_snapshot,
                )))
                .bind(base_snapshot)
                .fetch_one(&mut *tx)
                .await?;
            }
            if appended_rows.is_none() && appended_inlined == 0 {
                return Ok(None);
            }

            sqlx::query(
                "UPDATE ducklake_data_file SET end_snapshot = ?
                 WHERE table_id = ? AND end_snapshot IS NULL AND begin_snapshot > ?",
            )
            .bind(snapshot_id)
            .bind(table_id)
            .bind(base_snapshot)
            .execute(&mut *tx)
            .await?;
            for (name, system) in &inlined {
                sqlx::query(AssertSqlSafe(format!(
                    "UPDATE {} SET {} = ? WHERE {} > ? AND {} IS NULL",
                    quote_ident(name),
                    system.end_snapshot,
                    system.begin_snapshot,
                    system.end_snapshot,
                )))
                .bind(snapshot_id)
                .bind(base_snapshot)
                .execute(&mut *tx)
                .await?;
            }

            // `ducklake_table_stats` is left as it is, as official's `DELETE`
            // leaves it: `record_count` stays the gross count of rows ever
            // inserted, and next_row_id is untouched.

            let changes_made = match (appended_rows.is_some(), appended_inlined > 0) {
                (true, true) => format!("deleted_from_table:{table_id},inlined_delete:{table_id}"),
                (false, true) => format!("inlined_delete:{table_id}"),
                _ => format!("deleted_from_table:{table_id}"),
            };
            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &changes_made,
                &SnapshotCommitMetadata::default(),
            )
            .await?;

            tx.commit().await?;
            Ok(Some(snapshot_id))
        })
    }

    fn commit_truncate(
        &self,
        table_id: i64,
        // SQLite created the schema/table at begin; names unused (trait parity).
        _schema_name: &str,
        _table_name: &str,
        base_snapshot: i64,
    ) -> Result<u64> {
        block_on(async {
            // Metadata-only truncate in one snapshot: end every live data file and
            // its live delete file (as drop_table does); `ducklake_table_stats` is
            // left as official's `DELETE` leaves it.
            let mut tx = self.pool.begin().await?;

            // Write-lock-first (carry schema_version forward — not DDL).
            let (snapshot_id, _schema_version) = insert_snapshot(&mut tx).await?;

            // No-op guard: if the table has no live data file there is nothing to
            // truncate. Return Ok(0) WITHOUT committing so `tx` (and the snapshot
            // row insert_snapshot just made) rolls back, leaving no trace — same
            // as `drop_table`'s idempotent early return. Prevents a content-free
            // snapshot per repeated `DELETE FROM t` when the catalog's pinned
            // snapshot still sees already-ended files as live.
            let has_live_data: Option<i64> = sqlx::query_scalar(
                "SELECT 1 FROM ducklake_data_file
                 WHERE table_id = ? AND end_snapshot IS NULL LIMIT 1",
            )
            .bind(table_id)
            .fetch_optional(&mut *tx)
            .await?;
            let live_inlined = live_inlined_row_count(&mut tx, table_id).await?;
            if has_live_data.is_none() && live_inlined == 0 {
                return Ok(0);
            }
            let ours = TableChanges {
                deleted_from: has_live_data.is_some(),
                deleted_inlined: live_inlined > 0,
                ..TableChanges::default()
            };
            check_table_conflicts(&mut tx, table_id, base_snapshot, snapshot_id, ours).await?;

            // Rows removed = gross record_count minus still-live delete counts,
            // computed BEFORE ending anything so it matches what we retire.
            let gross: Option<i64> = sqlx::query_scalar(
                "SELECT COALESCE(SUM(record_count), 0) FROM ducklake_data_file WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(table_id)
            .fetch_optional(&mut *tx)
            .await?;
            let deleted: i64 = sqlx::query_scalar(
                "SELECT COALESCE(SUM(delete_count), 0) FROM ducklake_delete_file
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(table_id)
            .fetch_one(&mut *tx)
            .await?;
            let inlined_table = crate::metadata_provider::inlined_delete_table_name(table_id)?;
            let inlined_exists: Option<i64> =
                sqlx::query_scalar("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?")
                    .bind(&inlined_table)
                    .fetch_optional(&mut *tx)
                    .await?;
            let inlined_deleted: i64 = if inlined_exists.is_some() {
                sqlx::query_scalar(AssertSqlSafe(format!(
                    "SELECT COUNT(*) FROM (
                       SELECT DISTINCT d.file_id, d.row_id
                       FROM \"{inlined_table}\" d
                       JOIN ducklake_data_file f ON f.data_file_id = d.file_id
                       WHERE f.table_id = ? AND f.end_snapshot IS NULL
                         AND d.begin_snapshot <= ?
                     )"
                )))
                .bind(table_id)
                .bind(snapshot_id)
                .fetch_one(&mut *tx)
                .await?
            } else {
                0
            };
            let live_rows =
                (gross.unwrap_or(0) + live_inlined - deleted - inlined_deleted).max(0) as u64;

            sqlx::query(
                "UPDATE ducklake_data_file SET end_snapshot = ?
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(snapshot_id)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            for (table_name, system) in inlined_tables(&mut tx, table_id).await? {
                let sql = format!(
                    "UPDATE {} SET {end} = ? WHERE {end} IS NULL",
                    quote_ident(&table_name),
                    end = system.end_snapshot,
                );
                sqlx::query(AssertSqlSafe(sql))
                    .bind(snapshot_id)
                    .execute(&mut *tx)
                    .await?;
            }
            sqlx::query(
                "UPDATE ducklake_delete_file SET end_snapshot = ?
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(snapshot_id)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            // Official's tokens for the storage the truncate ended: Parquet
            // files, inlined rows, or both.
            let changes_made = crate::metadata_writer::table_storage_changes(
                table_id,
                None,
                ours.deleted_from,
                ours.deleted_inlined,
            );
            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &changes_made,
                &SnapshotCommitMetadata::default(),
            )
            .await?;

            tx.commit().await?;
            Ok(live_rows)
        })
    }

    fn publish_snapshot(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        _snapshot_id: i64,
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
    ) -> Result<CommitIds> {
        // Fileless commit point. Single-catalog SQLite defers the snapshot-row
        // insert out of begin_write_transaction, so this is no longer a no-op:
        // it inserts the deferred snapshot row + column generation and, for
        // Replace, retires the prior generation — making the new head visible
        // atomically. Used by CREATE TABLE (schema.rs); the crate's own write
        // path always registers a file (even for zero rows) and so commits via
        // register_data_file instead.
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let snapshot_id = finalize_snapshot(
                &mut tx,
                table_id,
                schema_name,
                table_name,
                columns,
                column_ids,
                mode,
                base_snapshot,
            )
            .await?;
            let replaced_existing_data = replaced_storage(&mut tx, table_id, snapshot_id).await?;
            let changes_made = table_write_changes(table_id, mode, false, replaced_existing_data);
            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &changes_made,
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            let schema_id: i64 =
                sqlx::query("SELECT schema_id FROM ducklake_table WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?
                    .try_get(0)?;
            tx.commit().await?;
            Ok(CommitIds {
                snapshot_id,
                schema_id,
                table_id,
            })
        })
    }

    fn end_table_files(&self, table_id: i64, snapshot_id: i64) -> Result<u64> {
        // Used by WriteMode::Replace. End-snapshotting every visible file
        // drops the table's currently-visible row count and byte total to
        // zero (the new files written next will rebuild them). `next_row_id`
        // is deliberately NOT reset: rowids must stay monotonic across the
        // table's lifetime so historical snapshots still resolve uniquely.
        block_on(async {
            let mut tx = self.pool.begin().await?;

            let result = sqlx::query(
                "UPDATE ducklake_data_file SET end_snapshot = ?
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(snapshot_id)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            tx.commit().await?;
            Ok(result.rows_affected())
        })
    }

    fn get_data_path(&self) -> Result<String> {
        block_on(async {
            let row =
                sqlx::query("SELECT value FROM ducklake_metadata WHERE key = ? AND scope IS NULL")
                    .bind("data_path")
                    .fetch_optional(&self.pool)
                    .await?;

            match row {
                Some(r) => Ok(r.try_get(0)?),
                None => Err(crate::error::DuckLakeError::InvalidConfig(
                    "Missing required catalog metadata: 'data_path' not configured.".to_string(),
                )),
            }
        })
    }

    fn get_table_column_nullability(
        &self,
        schema_name: &str,
        table_name: &str,
    ) -> Result<Option<Vec<(String, bool)>>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT c.column_name, c.nulls_allowed
                 FROM ducklake_column c
                 JOIN ducklake_table t ON t.table_id = c.table_id
                 JOIN ducklake_schema s ON s.schema_id = t.schema_id
                 WHERE s.schema_name = ? AND s.end_snapshot IS NULL
                   AND t.table_name = ? AND t.end_snapshot IS NULL
                   AND c.end_snapshot IS NULL AND c.parent_column IS NULL
                 ORDER BY c.column_order",
            )
            .bind(schema_name)
            .bind(table_name)
            .fetch_all(&self.pool)
            .await?;
            if rows.is_empty() {
                return Ok(None);
            }
            rows.into_iter()
                .map(|row| {
                    let name: String = row.try_get(0)?;
                    let nullable = row.try_get::<Option<bool>, _>(1)?.unwrap_or(true);
                    Ok((name, nullable))
                })
                .collect::<std::result::Result<Vec<_>, sqlx::Error>>()
                .map(Some)
                .map_err(Into::into)
        })
    }

    fn set_data_path(&self, path: &str) -> Result<()> {
        let path = directory_path(path);
        block_on(async {
            let current: Option<String> = sqlx::query_scalar(
                "SELECT value FROM ducklake_metadata WHERE key = 'data_path' AND scope IS NULL",
            )
            .fetch_optional(&self.pool)
            .await?;
            if current.as_deref() == Some(path.as_str()) {
                return Ok(());
            }

            sqlx::query("DELETE FROM ducklake_metadata WHERE key = 'data_path' AND scope IS NULL")
                .execute(&self.pool)
                .await?;

            sqlx::query(
                "INSERT INTO ducklake_metadata (key, value, scope)
                 VALUES ('data_path', ?, NULL)",
            )
            .bind(&path)
            .execute(&self.pool)
            .await?;

            Ok(())
        })
    }

    fn initialize_schema(&self) -> Result<()> {
        block_on(async {
            sqlx::query(SQL_CREATE_SCHEMA).execute(&self.pool).await?;
            sqlx::query(SQL_CREATE_INLINED_DATA_TABLES)
                .execute(&self.pool)
                .await?;
            sqlx::query(SQL_CREATE_TAG_TABLES)
                .execute(&self.pool)
                .await?;
            // Upgrade a pre-existing catalog's `ducklake_column` from the legacy
            // single-row-PK shape to upstream's bare shape (idempotent, crash-safe).
            // `CREATE TABLE IF NOT EXISTS` above only shapes new catalogs.
            migrate_ducklake_column_drop_pk(&self.pool).await?;
            migrate_add_metadata_scope_columns(&self.pool).await?;
            // Upgrade a pre-existing catalog to track schema_version (add the
            // ducklake_snapshot.schema_version column; idempotent, lossless).
            migrate_add_schema_version(&self.pool).await?;
            // Upgrade a pre-existing catalog to the v1.0 partial-file marker (add
            // partial_max to ducklake_data_file and ducklake_delete_file;
            // idempotent, lossless — NULL means "not a partial file").
            migrate_add_partial_max(&self.pool).await?;
            // Upgrade a pre-existing catalog to carry ducklake_data_file.partition_id
            // (idempotent, lossless — NULL means "not partitioned").
            migrate_add_partition_id(&self.pool).await?;
            migrate_snapshot_changes_nullable(&self.pool).await?;
            // Seed the monotonic id allocators. snapshot_id and column_id are
            // reserved in begin_write_transaction and inserted at the commit, so
            // they can't use rowid autoincrement (inserting the ducklake_snapshot
            // row would advance the head before the data is committed). The
            // counters give collision-free allocation across concurrent writers.
            // Idempotent on re-open, and seeded from the current MAX so a
            // pre-existing catalog continues without reusing ids.
            sqlx::query(
                "INSERT INTO ducklake_metadata (key, value, scope)
                 SELECT 'next_column_id',
                        CAST(COALESCE((SELECT MAX(column_id) FROM ducklake_column), 0) AS TEXT),
                        NULL
                 WHERE NOT EXISTS (
                     SELECT 1 FROM ducklake_metadata WHERE key = 'next_column_id' AND scope IS NULL
                 )",
            )
            .execute(&self.pool)
            .await?;
            // Seed the partition-id allocator (mirrors next_column_id). Reserved via
            // reserve_ids at set_partition_spec time. Seeded from the current MAX so a
            // pre-existing catalog continues without reusing partition ids.
            sqlx::query(
                "INSERT INTO ducklake_metadata (key, value, scope)
                 SELECT 'next_partition_id',
                        CAST(COALESCE((SELECT MAX(partition_id) FROM ducklake_partition_info), 0) AS TEXT),
                        NULL
                 WHERE NOT EXISTS (
                     SELECT 1 FROM ducklake_metadata WHERE key = 'next_partition_id' AND scope IS NULL
                 )",
            )
            .execute(&self.pool)
            .await?;
            // Seed the sort-id allocator (mirrors next_partition_id). Reserved via
            // reserve_ids at set_sort_spec time. Seeded from the current MAX so a
            // pre-existing catalog continues without reusing sort ids.
            sqlx::query(
                "INSERT INTO ducklake_metadata (key, value, scope)
                 SELECT 'next_sort_id',
                        CAST(COALESCE((SELECT MAX(sort_id) FROM ducklake_sort_info), 0) AS TEXT),
                        NULL
                 WHERE NOT EXISTS (
                     SELECT 1 FROM ducklake_metadata WHERE key = 'next_sort_id' AND scope IS NULL
                 )",
            )
            .execute(&self.pool)
            .await?;
            migrate_directory_paths(&self.pool).await?;
            Ok(())
        })
    }

    fn begin_write_transaction(
        &self,
        schema_name: &str,
        table_name: &str,
        columns: &[ColumnDef],
        mode: WriteMode,
    ) -> Result<WriteSetupResult> {
        validate_name(schema_name, "Schema")?;
        validate_name(table_name, "Table")?;
        if columns.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "Table must have at least one column".to_string(),
            ));
        }
        block_on(async {
            let mut tx = self.pool.begin().await?;

            // Reserve the column ids first so the counter UPDATE takes the write
            // lock up front, avoiding a lock-upgrade deadlock between concurrent
            // begins. These ids match the staged parquet field ids.
            let catalog_columns = catalog_column_defs(columns)?;
            let n = catalog_columns.len() as i64;
            let last_column_id = reserve_ids(&mut tx, "next_column_id", n).await?;
            // Freshly reserved ids. Only a genuinely-new column actually consumes
            // one below; an existing column keeps its current id, so some of these
            // may go unused (harmless monotonic-counter gaps).
            let fresh_ids: Vec<i64> = ((last_column_id - n + 1)..=last_column_id).collect();

            // The catalog head this write is based on; a Replace commit aborts if
            // another writer published a newer generation of the table past it.
            let base_snapshot_id: i64 =
                sqlx::query("SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot")
                    .fetch_one(&mut *tx)
                    .await?
                    .try_get(0)?;

            // Tentative id for WriteSetupResult; the real one is assigned at the
            // commit (finalize_snapshot), so it may differ under concurrency.
            let snapshot_id: i64 = base_snapshot_id + 1;

            let schema_id: i64 = {
                let existing = sqlx::query(
                    "SELECT schema_id FROM ducklake_schema
                     WHERE schema_name = ? AND end_snapshot IS NULL",
                )
                .bind(schema_name)
                .fetch_optional(&mut *tx)
                .await?;

                if let Some(row) = existing {
                    row.try_get(0)?
                } else {
                    let row = sqlx::query(
                        "INSERT INTO ducklake_schema (schema_name, path, path_is_relative, begin_snapshot)
                         VALUES (?, ?, 1, ?) RETURNING schema_id",
                    )
                    .bind(schema_name)
                    .bind(directory_path(schema_name))
                    .bind(snapshot_id)
                    .fetch_one(&mut *tx)
                    .await?;
                    row.try_get(0)?
                }
            };

            let table_id: i64 = {
                let existing = sqlx::query(
                    "SELECT table_id FROM ducklake_table
                     WHERE schema_id = ? AND table_name = ? AND end_snapshot IS NULL",
                )
                .bind(schema_id)
                .bind(table_name)
                .fetch_optional(&mut *tx)
                .await?;

                if let Some(row) = existing {
                    row.try_get(0)?
                } else {
                    let row = sqlx::query(
                        "INSERT INTO ducklake_table (schema_id, table_name, path, path_is_relative, begin_snapshot)
                         VALUES (?, ?, ?, 1, ?) RETURNING table_id",
                    )
                    .bind(schema_id)
                    .bind(table_name)
                    .bind(directory_path(table_name))
                    .bind(snapshot_id)
                    .fetch_one(&mut *tx)
                    .await?;
                    row.try_get(0)?
                }
            };

            // Get existing columns to (a) check schema compatibility for appends
            // and (b) REUSE each column's id. column_id == parquet field_id == a
            // column's stable identity; re-minting it every write would orphan the
            // field_ids already baked into previously-written files, so an
            // unchanged column must keep its id.
            let rows = sqlx::query(
                "SELECT column_name, column_type, nulls_allowed, column_id, parent_column
                 FROM ducklake_column
                 WHERE table_id = ? AND end_snapshot IS NULL
                 ORDER BY column_order",
            )
            .bind(table_id)
            .fetch_all(&mut *tx)
            .await?;

            let mut existing_catalog_columns = Vec::with_capacity(rows.len());
            for row in rows {
                let name: String = row.try_get(0)?;
                let col_type: String = row.try_get(1)?;
                let _nullable: bool = row.try_get::<Option<bool>, _>(2)?.unwrap_or(true);
                let cid: i64 = row.try_get(3)?;
                let parent_column: Option<i64> = row.try_get(4)?;
                existing_catalog_columns.push(ExistingCatalogColumn {
                    column_id: cid,
                    name,
                    ducklake_type: col_type,
                    parent_column,
                });
            }

            let field_ids =
                assign_column_ids(&catalog_columns, &existing_catalog_columns, &fresh_ids)?;

            // Data-write policy (§5): a data write — Replace OR Append — must NOT
            // change a column's type. A type change is schema evolution and must
            // go through `promote_column_type`, never a data write; silently
            // keeping the old catalog type (the historic "C" bug) corrupts reads.
            // The comparison is canonical (`int64` ≡ `bigint`) so an alias-only
            // restatement is a no-op, not an error. Append additionally requires a
            // genuinely new column to be nullable (a Replace overwrites every row,
            // so a new non-nullable column is fine there).
            if !existing_catalog_columns.is_empty() {
                use std::collections::HashMap;

                let existing_map: HashMap<i64, &ExistingCatalogColumn> = existing_catalog_columns
                    .iter()
                    .map(|column| (column.column_id, column))
                    .collect();

                for (new_column, column_id) in catalog_columns.iter().zip(&field_ids) {
                    if let Some(existing_column) = existing_map.get(column_id) {
                        let same_type =
                            catalog_column_type_equal(&existing_column.ducklake_type, new_column);
                        if !same_type {
                            return Err(crate::error::DuckLakeError::UnsupportedTypeChange {
                                operation: TypeChangeOperation::DataWrite {
                                    mode: match mode {
                                        WriteMode::Replace => TypeChangeWriteMode::Replace,
                                        WriteMode::Append => TypeChangeWriteMode::Append,
                                    },
                                },
                                column: new_column.name.clone(),
                                from: existing_column.ducklake_type.clone(),
                                to: new_column.ducklake_type.clone(),
                            });
                        }
                    } else if mode == WriteMode::Append
                        && new_column.parent_index.is_none()
                        && !new_column.is_nullable
                    {
                        return Err(crate::error::DuckLakeError::InvalidConfig(format!(
                            "Schema evolution error: new column '{}' must be nullable. Adding non-nullable columns is not allowed.",
                            new_column.name
                        )));
                    }
                }
            }

            // Final per-column ids: reuse the existing id for a column already in
            // the table, consume a freshly reserved id only for a genuinely new
            // column. These are baked into the staged parquet's field_id metadata,
            // so they must equal the ids `finalize_snapshot` commits. Column rows
            // themselves are written at the commit point (not here): the SQLite
            // read path resolves columns by `end_snapshot IS NULL` only (not
            // snapshot-scoped), so inserting at begin would leak the new
            // generation to concurrent reads during the upload window.
            // No snapshot row, no column rows, and no Replace retirement are
            // written here — all are deferred to the atomic commit so the head
            // never resolves to an incomplete snapshot. TX-A commits only the
            // idempotent get-or-create schema/table rows; they carry
            // begin_snapshot = the reserved id and stay invisible until the
            // snapshot publishes, since schema/table reads ARE snapshot-scoped.
            tx.commit().await?;

            Ok(WriteSetupResult {
                snapshot_id,
                base_snapshot_id,
                schema_id,
                table_id,
                column_ids: top_level_column_ids(&catalog_columns, &field_ids)?,
                field_ids,
            })
        })
    }
}

async fn enforce_inline_partition_fence(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
    base_snapshot: i64,
) -> Result<()> {
    let changed: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_partition_info WHERE table_id = ? AND (begin_snapshot > ? OR end_snapshot > ?)")
        .bind(table_id)
        .bind(base_snapshot)
        .bind(base_snapshot)
        .fetch_one(&mut **tx).await?;
    if changed != 0 {
        return Err(crate::DuckLakeError::Conflict(format!(
            "partition spec for table {table_id} changed during inline write; retry the write"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DuckLakeCatalog;
    use crate::metadata_provider::MetadataProvider;
    use arrow::array::Int64Array;
    use datafusion::prelude::SessionContext;
    use tempfile::TempDir;

    #[test]
    fn sqlite_inlined_types_follow_ducklake_encodings() {
        assert_eq!(inlined_sqlite_type(&DataType::Int32), "BIGINT");
        assert_eq!(inlined_sqlite_type(&DataType::UInt64), "VARCHAR");
        assert_eq!(inlined_sqlite_type(&DataType::Float64), "VARCHAR");
        assert_eq!(inlined_sqlite_type(&DataType::Float32), "VARCHAR");
        assert_eq!(inlined_sqlite_type(&DataType::Binary), "BLOB");
        assert_eq!(inlined_sqlite_type(&DataType::BinaryView), "BLOB");
        assert_eq!(
            inlined_sqlite_type(&DataType::FixedSizeBinary(16)),
            "VARCHAR"
        );
        assert_eq!(inlined_sqlite_type(&DataType::FixedSizeBinary(32)), "BLOB");
        assert_eq!(inlined_sqlite_type(&DataType::Date32), "VARCHAR");
        assert_eq!(
            inlined_sqlite_type(&DataType::Timestamp(
                arrow::datatypes::TimeUnit::Nanosecond,
                None,
            )),
            "VARCHAR"
        );
    }

    async fn create_test_writer() -> (SqliteMetadataWriter, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());
        let writer = SqliteMetadataWriter::new_with_init(&conn_str)
            .await
            .unwrap();
        (writer, temp_dir)
    }

    #[rstest::rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn staged_inline_changes_follow_storage_and_share_snapshot() {
        let (writer, temp) = create_test_writer().await;
        writer.set_data_path(temp.path().to_str().unwrap()).unwrap();
        let columns = vec![ColumnDef::new("value", "BIGINT", false).unwrap()];
        let batch = RecordBatch::try_from_iter(vec![(
            "value",
            Arc::new(Int64Array::from(vec![31, 47])) as Arc<dyn Array>,
        )])
        .unwrap();
        let mut writes = Vec::new();
        for (name, data) in [
            (
                "files",
                StagedTableData::Files(vec![DataFileInfo::new("rows.parquet", 123, 3)]),
            ),
            ("inline", StagedTableData::Inlined(vec![batch.clone()])),
        ] {
            let setup = writer
                .begin_write_transaction("main", name, &columns, WriteMode::Append)
                .unwrap();
            writes.push(StagedTableWrite {
                table_id: setup.table_id,
                schema_name: "main".to_string(),
                table_name: name.to_string(),
                base_snapshot_id: setup.base_snapshot_id,
                mode: WriteMode::Append,
                columns: columns.clone(),
                column_ids: setup.column_ids,
                data,
                snapshot_id_columns: Vec::new(),
                positional_deletes: Vec::new(),
                inlined_deletes: Vec::new(),
                inlined_row_ids: None,
                inlined_file_deletes: Vec::new(),
                rewritten_deletes: Vec::new(),
                dropped_files: Vec::new(),
                inlined_flush: false,
            });
        }
        let commit = writer
            .commit_multi_table(&writes, &SnapshotCommitMetadata::default(), None)
            .unwrap();
        let provider = crate::SqliteMetadataProvider::new(&format!(
            "sqlite:{}",
            temp.path().join("test.db").display()
        ))
        .await
        .unwrap();
        let changes = provider.list_snapshot_changes().unwrap();
        let changes = changes
            .iter()
            .find(|change| change.snapshot_id == commit.snapshot_id)
            .unwrap()
            .changes_made
            .as_deref()
            .unwrap();
        assert_eq!(
            changes,
            format!(
                "created_schema:\"main\",created_table:\"main\".\"files\",created_table:\"main\".\"inline\",inserted_into_table:{},inlined_insert:{}",
                writes[0].table_id, writes[1].table_id
            )
        );
        let table_columns = provider
            .get_table_structure(writes[1].table_id, commit.snapshot_id)
            .unwrap();
        let inline = provider
            .get_inlined_data_with_row_ids(writes[1].table_id, commit.snapshot_id, &table_columns)
            .unwrap();
        assert_eq!(inline.len(), 1);
        assert_eq!(inline[0].row_ids, vec![0, 1]);
        assert_eq!(inline[0].begin_snapshots, vec![commit.snapshot_id; 2]);
        assert_eq!(inline[0].batch, batch);
        let file_snapshot: i64 =
            sqlx::query_scalar("SELECT begin_snapshot FROM ducklake_data_file WHERE table_id = ?")
                .bind(writes[0].table_id)
                .fetch_one(&writer.pool)
                .await
                .unwrap();
        assert_eq!(file_snapshot, commit.snapshot_id);

        let mut replacement = writes[1].clone();
        replacement.base_snapshot_id = commit.snapshot_id;
        replacement.mode = WriteMode::Replace;
        let replaced = writer
            .commit_multi_table(
                &[replacement.clone()],
                &SnapshotCommitMetadata::default(),
                Some(commit.snapshot_id),
            )
            .unwrap();
        let changes = provider.list_snapshot_changes().unwrap();
        assert_eq!(
            changes.last().unwrap().changes_made,
            Some(format!(
                "inlined_insert:{},inlined_delete:{}",
                replacement.table_id, replacement.table_id
            ))
        );
        let current = provider
            .get_inlined_data_with_row_ids(
                replacement.table_id,
                replaced.snapshot_id,
                &table_columns,
            )
            .unwrap();
        assert_eq!(current[0].row_ids, vec![2, 3]);
        replacement.mode = WriteMode::Append;
        replacement.base_snapshot_id = replaced.snapshot_id;
        replacement.data = StagedTableData::None;
        replacement.inlined_deletes = vec![InlinedRowRef {
            table_name: current[0].table_name.clone(),
            row_id: 2,
        }];
        let deleted = writer
            .commit_multi_table(
                &[replacement],
                &SnapshotCommitMetadata::default(),
                Some(replaced.snapshot_id),
            )
            .unwrap();
        let remaining = provider
            .get_inlined_data_with_row_ids(writes[1].table_id, deleted.snapshot_id, &table_columns)
            .unwrap();
        assert_eq!(remaining[0].row_ids, vec![3]);
        assert_eq!(
            remaining[0]
                .batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values(),
            &[47]
        );
        assert_eq!(
            provider
                .list_snapshot_changes()
                .unwrap()
                .last()
                .unwrap()
                .changes_made,
            Some(format!("inlined_delete:{}", writes[1].table_id))
        );
        let gross: i64 =
            sqlx::query_scalar("SELECT record_count FROM ducklake_table_stats WHERE table_id = ?")
                .bind(writes[1].table_id)
                .fetch_one(&writer.pool)
                .await
                .unwrap();
        // Every row ever inserted: the first write's two and the Replace's two.
        assert_eq!(gross, 4);
        let mut update = writes[1].clone();
        update.base_snapshot_id = deleted.snapshot_id;
        update.data = StagedTableData::Inlined(vec![
            RecordBatch::try_from_iter(vec![(
                "value",
                Arc::new(Int64Array::from(vec![59])) as Arc<dyn Array>,
            )])
            .unwrap(),
        ]);
        update.inlined_deletes = [3, 3, 4, 999]
            .into_iter()
            .map(|row_id| InlinedRowRef {
                table_name: remaining[0].table_name.clone(),
                row_id,
            })
            .collect();
        let updated = writer
            .commit_multi_table(
                &[update],
                &SnapshotCommitMetadata::default(),
                Some(deleted.snapshot_id),
            )
            .unwrap();
        let latest = provider
            .get_inlined_data_with_row_ids(writes[1].table_id, updated.snapshot_id, &table_columns)
            .unwrap();
        let gross: i64 =
            sqlx::query_scalar("SELECT record_count FROM ducklake_table_stats WHERE table_id = ?")
                .bind(writes[1].table_id)
                .fetch_one(&writer.pool)
                .await
                .unwrap();
        assert_eq!(latest[0].row_ids, vec![4]);
        assert_eq!(
            latest[0]
                .batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values(),
            &[59]
        );
        // The update's new row is one more insert.
        assert_eq!(gross, 5);
        let mut stale = writes[1].clone();
        stale.base_snapshot_id = deleted.snapshot_id;
        stale.data = StagedTableData::None;
        stale.inlined_deletes = vec![InlinedRowRef {
            table_name: remaining[0].table_name.clone(),
            row_id: 3,
        }];
        let error = writer
            .commit_multi_table(&[stale], &SnapshotCommitMetadata::default(), None)
            .unwrap_err();
        assert!(matches!(error, crate::DuckLakeError::Conflict(_)));
        assert_eq!(
            provider.get_current_snapshot().unwrap(),
            updated.snapshot_id
        );
        let ctx = SessionContext::new();
        ctx.register_catalog("dl", Arc::new(DuckLakeCatalog::new(provider).unwrap()));
        let actual = ctx
            .sql("SELECT COUNT(*) FROM dl.main.inline")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(
            actual[0]
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values(),
            &[1]
        );
    }

    /// An existing user's catalog (legacy `column_id INTEGER PRIMARY KEY`) must be
    /// upgraded in place to upstream's bare shape: rows + `column_id`s preserved,
    /// the single-row PK gone (so versioned/promoted columns become expressible),
    /// and the migration idempotent. This is the "people already using the
    /// library" path — `CREATE TABLE IF NOT EXISTS` alone would not touch them.
    #[tokio::test(flavor = "multi_thread")]
    async fn migrate_legacy_column_pk_to_bare_shape() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("legacy.db");
        let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());
        let pool = SqlitePool::connect(&conn_str).await.unwrap();

        // Build the LEGACY ducklake_column exactly as a pre-versioning catalog had it.
        sqlx::query(
            "CREATE TABLE ducklake_column (
                column_id INTEGER PRIMARY KEY,
                table_id INTEGER NOT NULL,
                column_name VARCHAR NOT NULL,
                column_type VARCHAR NOT NULL,
                column_order INTEGER NOT NULL,
                nulls_allowed BOOLEAN DEFAULT 1,
                initial_default VARCHAR,
                default_value VARCHAR,
                parent_column INTEGER,
                default_value_type VARCHAR,
                default_value_dialect VARCHAR,
                begin_snapshot INTEGER NOT NULL,
                end_snapshot INTEGER
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO ducklake_column
                 (column_id, table_id, column_name, column_type, column_order, nulls_allowed, begin_snapshot)
             VALUES (5, 1, 'id', 'int32', 0, 1, 1)",
        )
        .execute(&pool)
        .await
        .unwrap();

        // Sanity: the legacy PK rejects a second row sharing the column_id.
        let dup = sqlx::query(
            "INSERT INTO ducklake_column
                 (column_id, table_id, column_name, column_type, column_order, nulls_allowed, begin_snapshot)
             VALUES (5, 1, 'id', 'int64', 0, 1, 2)",
        )
        .execute(&pool)
        .await;
        assert!(
            dup.is_err(),
            "legacy single-row PK must reject a duplicate column_id"
        );

        // Migrate.
        migrate_ducklake_column_drop_pk(&pool).await.unwrap();

        // Row + column_id value preserved.
        let row = sqlx::query(
            "SELECT column_id, column_name, column_type FROM ducklake_column WHERE begin_snapshot = 1",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(row.try_get::<i64, _>("column_id").unwrap(), 5);
        assert_eq!(row.try_get::<String, _>("column_name").unwrap(), "id");
        assert_eq!(row.try_get::<String, _>("column_type").unwrap(), "int32");

        // The PK is gone: a SECOND version row with the same column_id now coexists
        // (the whole point — versioned / type-promoted columns).
        sqlx::query(
            "INSERT INTO ducklake_column
                 (column_id, begin_snapshot, end_snapshot, table_id, column_order, column_name, column_type, nulls_allowed)
             VALUES (5, 2, NULL, 1, 0, 'id', 'int64', 1)",
        )
        .execute(&pool)
        .await
        .unwrap();
        let cnt: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_column WHERE column_id = 5")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(
            cnt, 2,
            "two version rows sharing a column_id must coexist post-migration"
        );

        // Idempotent: re-running is a no-op and leaves data intact.
        migrate_ducklake_column_drop_pk(&pool).await.unwrap();
        let cnt2: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_column")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(cnt2, 2, "migration must be idempotent");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn migrate_bare_column_table_adds_default_metadata() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("legacy-defaults.db");
        let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());
        let pool = SqlitePool::connect(&conn_str).await.unwrap();

        sqlx::query(
            "CREATE TABLE ducklake_column (
                column_id BIGINT,
                begin_snapshot BIGINT,
                end_snapshot BIGINT,
                table_id BIGINT,
                column_order BIGINT,
                column_name VARCHAR,
                column_type VARCHAR,
                nulls_allowed BOOLEAN,
                parent_column BIGINT
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO ducklake_column
             (column_id, begin_snapshot, table_id, column_order, column_name,
              column_type, nulls_allowed)
             VALUES (5, 1, 1, 0, 'id', 'int32', 1)",
        )
        .execute(&pool)
        .await
        .unwrap();

        migrate_ducklake_column_drop_pk(&pool).await.unwrap();

        let default_columns: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM pragma_table_info('ducklake_column')
             WHERE name IN ('initial_default', 'default_value',
                            'default_value_type', 'default_value_dialect')",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(default_columns, 4);
        let row: (i64, String) =
            sqlx::query_as("SELECT column_id, column_name FROM ducklake_column WHERE table_id = 1")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(row, (5, "id".to_string()));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn migrate_snapshot_changes_to_nullable() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("legacy.db");
        let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());
        let pool = SqlitePool::connect(&conn_str).await.unwrap();
        sqlx::query(
            "CREATE TABLE ducklake_snapshot_changes (
                snapshot_id INTEGER PRIMARY KEY,
                changes_made VARCHAR NOT NULL,
                author VARCHAR,
                commit_message VARCHAR,
                commit_extra_info VARCHAR
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO ducklake_snapshot_changes
                 (snapshot_id, changes_made, author, commit_message, commit_extra_info)
             VALUES (1, '', NULL, NULL, NULL),
                    (2, 'inserted_into_table:7', 'author', 'message', 'extra')",
        )
        .execute(&pool)
        .await
        .unwrap();

        migrate_snapshot_changes_nullable(&pool).await.unwrap();

        let not_null: i64 = sqlx::query_scalar(
            "SELECT `notnull` FROM pragma_table_info('ducklake_snapshot_changes')
             WHERE name = 'changes_made'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        let rows = sqlx::query_as::<
            _,
            (
                i64,
                Option<String>,
                Option<String>,
                Option<String>,
                Option<String>,
            ),
        >(
            "SELECT snapshot_id, changes_made, author, commit_message, commit_extra_info
             FROM ducklake_snapshot_changes ORDER BY snapshot_id",
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(not_null, 0);
        assert_eq!(
            rows,
            vec![
                (1, None, None, None, None),
                (
                    2,
                    Some("inserted_into_table:7".to_string()),
                    Some("author".to_string()),
                    Some("message".to_string()),
                    Some("extra".to_string()),
                ),
            ],
        );

        migrate_snapshot_changes_nullable(&pool).await.unwrap();
        let row_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_snapshot_changes")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(row_count, 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn open_legacy_snapshot_changes_with_plain_new() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("legacy.db");
        let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());
        let writer = SqliteMetadataWriter::new_with_init(&conn_str)
            .await
            .unwrap();

        sqlx::query("DROP TABLE ducklake_snapshot_changes")
            .execute(&writer.pool)
            .await
            .unwrap();
        sqlx::query(
            "CREATE TABLE ducklake_snapshot_changes (
                snapshot_id INTEGER PRIMARY KEY,
                changes_made VARCHAR NOT NULL
            )",
        )
        .execute(&writer.pool)
        .await
        .unwrap();
        drop(writer);

        let writer = SqliteMetadataWriter::new(&conn_str).await.unwrap();
        let snapshot_id = writer.create_snapshot().unwrap();
        let changes_made: Option<String> = sqlx::query_scalar(
            "SELECT changes_made FROM ducklake_snapshot_changes WHERE snapshot_id = ?",
        )
        .bind(snapshot_id)
        .fetch_one(&writer.pool)
        .await
        .unwrap();
        assert_eq!(changes_made, None);

        let commit_metadata = SnapshotCommitMetadata::new()
            .with_author("Jane Doe")
            .with_message("Imported official catalog")
            .with_extra_info("migration-regression");
        let mut tx = writer.pool.begin().await.unwrap();
        record_snapshot_changes(
            &mut tx,
            snapshot_id,
            "inserted_into_table:7",
            &commit_metadata,
        )
        .await
        .unwrap();
        tx.commit().await.unwrap();

        let columns = sqlx::query(
            "SELECT name, `notnull` FROM pragma_table_info('ducklake_snapshot_changes')",
        )
        .fetch_all(&writer.pool)
        .await
        .unwrap()
        .into_iter()
        .map(|row| {
            (
                row.try_get::<String, _>("name").unwrap(),
                row.try_get::<i64, _>("notnull").unwrap(),
            )
        })
        .collect::<Vec<_>>();
        let row = sqlx::query_as::<
            _,
            (
                Option<String>,
                Option<String>,
                Option<String>,
                Option<String>,
            ),
        >(
            "SELECT changes_made, author, commit_message, commit_extra_info
             FROM ducklake_snapshot_changes WHERE snapshot_id = ?",
        )
        .bind(snapshot_id)
        .fetch_one(&writer.pool)
        .await
        .unwrap();

        assert_eq!(
            columns,
            vec![
                ("snapshot_id".to_string(), 0),
                ("changes_made".to_string(), 0),
                ("author".to_string(), 0),
                ("commit_message".to_string(), 0),
                ("commit_extra_info".to_string(), 0),
            ],
        );
        assert_eq!(
            row,
            (
                Some("inserted_into_table:7".to_string()),
                Some("Jane Doe".to_string()),
                Some("Imported official catalog".to_string()),
                Some("migration-regression".to_string()),
            ),
        );
    }

    /// An existing catalog whose `ducklake_snapshot` predates `schema_version`
    /// (the pre-#151 shape) must gain the column in place: idempotent, lossless,
    /// and existing snapshot rows take the `DEFAULT 0`. `CREATE TABLE IF NOT
    /// EXISTS` alone would not touch a pre-existing catalog.
    #[tokio::test(flavor = "multi_thread")]
    async fn migrate_add_schema_version_to_legacy_snapshot() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("legacy.db");
        let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());
        let pool = SqlitePool::connect(&conn_str).await.unwrap();

        // Legacy ducklake_snapshot: no schema_version column, with existing rows.
        sqlx::query(
            "CREATE TABLE ducklake_snapshot (
                snapshot_id INTEGER PRIMARY KEY,
                snapshot_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query("INSERT INTO ducklake_snapshot (snapshot_id) VALUES (1), (2)")
            .execute(&pool)
            .await
            .unwrap();

        // Migrate.
        migrate_add_schema_version(&pool).await.unwrap();

        // Column now exists and existing rows are backfilled to 0 (lossless).
        let has_col: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM pragma_table_info('ducklake_snapshot') WHERE name = 'schema_version'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(has_col, 1, "schema_version column must be added");
        let max_v: i64 =
            sqlx::query_scalar("SELECT COALESCE(MAX(schema_version), 0) FROM ducklake_snapshot")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(
            max_v, 0,
            "pre-existing snapshots backfill to schema_version 0"
        );
        let rows: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_snapshot")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(rows, 2, "no snapshot rows lost in the migration");

        // Idempotent: re-running is a no-op.
        migrate_add_schema_version(&pool).await.unwrap();
        let rows2: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_snapshot")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(rows2, 2, "migration must be idempotent");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_create_snapshot() {
        let (writer, _temp) = create_test_writer().await;

        let snap1 = writer.create_snapshot().unwrap();
        assert_eq!(snap1, 1);

        let snap2 = writer.create_snapshot().unwrap();
        assert_eq!(snap2, 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_or_create_schema() {
        let (writer, _temp) = create_test_writer().await;
        let snapshot_id = writer.create_snapshot().unwrap();

        // Create new schema
        let (schema_id, created) = writer
            .get_or_create_schema("main", None, snapshot_id)
            .unwrap();
        assert!(created);
        assert_eq!(schema_id, 1);

        // Get existing schema
        let (schema_id2, created2) = writer
            .get_or_create_schema("main", None, snapshot_id)
            .unwrap();
        assert!(!created2);
        assert_eq!(schema_id2, 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_or_create_table() {
        let (writer, _temp) = create_test_writer().await;
        let snapshot_id = writer.create_snapshot().unwrap();
        let (schema_id, _) = writer
            .get_or_create_schema("main", None, snapshot_id)
            .unwrap();

        // Create new table
        let (table_id, created) = writer
            .get_or_create_table(schema_id, "users", None, snapshot_id)
            .unwrap();
        assert!(created);
        assert_eq!(table_id, 1);

        // Get existing table
        let (table_id2, created2) = writer
            .get_or_create_table(schema_id, "users", None, snapshot_id)
            .unwrap();
        assert!(!created2);
        assert_eq!(table_id2, 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_set_columns() {
        let (writer, _temp) = create_test_writer().await;
        let snapshot_id = writer.create_snapshot().unwrap();
        let (schema_id, _) = writer
            .get_or_create_schema("main", None, snapshot_id)
            .unwrap();
        let (table_id, _) = writer
            .get_or_create_table(schema_id, "users", None, snapshot_id)
            .unwrap();

        let columns = vec![
            ColumnDef::new("id", "int64", false).unwrap(),
            ColumnDef::new("name", "varchar", true).unwrap(),
        ];

        let column_ids = writer.set_columns(table_id, &columns, snapshot_id).unwrap();
        assert_eq!(column_ids.len(), 2);
        assert_eq!(column_ids[0], 1);
        assert_eq!(column_ids[1], 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ddl_snapshots_record_spec_change_tokens() {
        let (writer, _temp) = create_test_writer().await;
        let bare_snapshot = writer.create_snapshot().unwrap();
        let ddl_snapshot = writer.create_snapshot().unwrap();
        let (schema_id, _) = writer
            .get_or_create_schema("ma\"in", None, ddl_snapshot)
            .unwrap();
        let (table_id, _) = writer
            .get_or_create_table(schema_id, "ev\"ents", None, ddl_snapshot)
            .unwrap();
        writer
            .set_columns(
                table_id,
                &[ColumnDef::new("id", "int64", false).unwrap()],
                ddl_snapshot,
            )
            .unwrap();
        let partition_snapshot = writer
            .set_partition_spec(
                table_id,
                &[("id".to_string(), PartitionTransform::Identity)],
            )
            .unwrap();
        let reset_snapshot = writer.reset_partition_spec(table_id).unwrap();

        let rows = sqlx::query_as::<_, (i64, Option<String>)>(
            "SELECT snapshot_id, changes_made
             FROM ducklake_snapshot_changes
             ORDER BY snapshot_id",
        )
        .fetch_all(&writer.pool)
        .await
        .unwrap();

        assert_eq!(
            rows,
            vec![
                (bare_snapshot, None),
                (
                    ddl_snapshot,
                    Some(
                        "created_schema:\"ma\"\"in\",created_table:\"ma\"\"in\".\"ev\"\"ents\""
                            .to_string(),
                    ),
                ),
                (
                    partition_snapshot,
                    Some(format!("altered_table:{table_id}"))
                ),
                (reset_snapshot, Some(format!("altered_table:{table_id}"))),
            ],
        );
    }

    /// Phase C (catalog-state faithfulness): a type promotion must leave the
    /// `ducklake_column` table in upstream DuckLake's exact versioned shape — TWO
    /// rows sharing the SAME `column_id`, the old one retired (`end_snapshot` set)
    /// and the new one live (`end_snapshot IS NULL`) with the widened type. This is
    /// precisely the two-rows-per-column the old single-row PK forbade.
    #[tokio::test(flavor = "multi_thread")]
    async fn promote_leaves_two_versioned_column_rows_same_id() {
        let (writer, _temp) = create_test_writer().await;
        let snap1 = writer.create_snapshot().unwrap();
        let (schema_id, _) = writer.get_or_create_schema("main", None, snap1).unwrap();
        let (table_id, _) = writer
            .get_or_create_table(schema_id, "t", None, snap1)
            .unwrap();
        writer
            .set_columns(
                table_id,
                &[ColumnDef::new("id", "int32", false).unwrap()],
                snap1,
            )
            .unwrap();

        let snap2 = writer.promote_column_type(table_id, "id", "int64").unwrap();
        assert!(snap2 > snap1, "promote creates a newer snapshot");

        let rows = sqlx::query(
            "SELECT column_id, column_type, begin_snapshot, end_snapshot
             FROM ducklake_column
             WHERE table_id = ? AND column_name = 'id'
             ORDER BY begin_snapshot",
        )
        .bind(table_id)
        .fetch_all(&writer.pool)
        .await
        .unwrap();

        assert_eq!(
            rows.len(),
            2,
            "promote must leave TWO versioned rows for the column"
        );

        let cid0: i64 = rows[0].try_get("column_id").unwrap();
        let type0: String = rows[0].try_get("column_type").unwrap();
        let end0: Option<i64> = rows[0].try_get("end_snapshot").unwrap();
        let cid1: i64 = rows[1].try_get("column_id").unwrap();
        let type1: String = rows[1].try_get("column_type").unwrap();
        let begin1: i64 = rows[1].try_get("begin_snapshot").unwrap();
        let end1: Option<i64> = rows[1].try_get("end_snapshot").unwrap();

        assert_eq!(
            cid0, cid1,
            "both versions share the SAME column_id (stable field-id)"
        );
        assert_eq!(type0, "int32", "old version retains its int32 type");
        assert_eq!(
            end0,
            Some(snap2),
            "old version retired at the promote snapshot"
        );
        assert_eq!(type1, "int64", "new version carries the widened int64 type");
        assert_eq!(begin1, snap2, "new version begins at the promote snapshot");
        assert_eq!(end1, None, "new version is the live one");

        // D4: exactly ONE live row per field-id after the promote.
        let live: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM ducklake_column
             WHERE table_id = ? AND column_name = 'id' AND end_snapshot IS NULL",
        )
        .bind(table_id)
        .fetch_one(&writer.pool)
        .await
        .unwrap();
        assert_eq!(
            live, 1,
            "exactly one live version per field-id after promote"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn top_level_name_resolution_ignores_nested_collision() {
        let (writer, _temp) = create_test_writer().await;
        let snapshot = writer.create_snapshot().unwrap();
        let (schema_id, _) = writer.get_or_create_schema("main", None, snapshot).unwrap();
        let (table_id, _) = writer
            .get_or_create_table(schema_id, "events", None, snapshot)
            .unwrap();
        writer
            .set_columns(
                table_id,
                &[ColumnDef::new("id", "int32", false).unwrap()],
                snapshot,
            )
            .unwrap();
        let top_level_id: i64 = sqlx::query_scalar(
            "SELECT column_id FROM ducklake_column
             WHERE table_id = ? AND column_name = 'id' AND parent_column IS NULL",
        )
        .bind(table_id)
        .fetch_one(&writer.pool)
        .await
        .unwrap();
        let nested_id = top_level_id + 1;
        sqlx::query(
            "INSERT INTO ducklake_column
                 (column_id, table_id, column_name, column_type, column_order,
                  nulls_allowed, parent_column, begin_snapshot)
             VALUES (?, ?, 'id', 'int16', 1, 1, ?, ?)",
        )
        .bind(nested_id)
        .bind(table_id)
        .bind(top_level_id)
        .bind(snapshot)
        .execute(&writer.pool)
        .await
        .unwrap();

        writer.promote_column_type(table_id, "id", "int64").unwrap();
        writer
            .set_partition_spec(
                table_id,
                &[("id".to_string(), PartitionTransform::Identity)],
            )
            .unwrap();
        writer
            .set_sort_spec(
                table_id,
                &[crate::sort::SortField::column(
                    0,
                    "id",
                    crate::sort::SortDirection::Asc,
                    crate::sort::NullOrder::NullsLast,
                )],
            )
            .unwrap();

        let top_level_type: String = sqlx::query_scalar(
            "SELECT column_type FROM ducklake_column
             WHERE table_id = ? AND column_name = 'id' AND parent_column IS NULL
               AND end_snapshot IS NULL",
        )
        .bind(table_id)
        .fetch_one(&writer.pool)
        .await
        .unwrap();
        let nested: (String, Option<i64>) = sqlx::query_as(
            "SELECT column_type, parent_column FROM ducklake_column
             WHERE table_id = ? AND column_id = ? AND end_snapshot IS NULL",
        )
        .bind(table_id)
        .bind(nested_id)
        .fetch_one(&writer.pool)
        .await
        .unwrap();
        let partition_column_id: i64 = sqlx::query_scalar(
            "SELECT column_id FROM ducklake_partition_column
             WHERE table_id = ? ORDER BY partition_id DESC LIMIT 1",
        )
        .bind(table_id)
        .fetch_one(&writer.pool)
        .await
        .unwrap();

        assert_eq!(top_level_type, "int64");
        assert_eq!(nested, ("int16".to_string(), Some(top_level_id)));
        assert_eq!(partition_column_id, top_level_id);
    }

    /// Phase D (old-version data → latest): a catalog whose `ducklake_column` is in
    /// the LEGACY single-row-PK shape (as a previous datafusion-ducklake version
    /// wrote it), holding real data files, must upgrade in place on re-open and read
    /// its values back intact — then support a promote. We simulate the old version
    /// by writing with the current writer, then rebuilding `ducklake_column` to the
    /// legacy shape; re-opening runs the forward migration.
    #[tokio::test(flavor = "multi_thread")]
    async fn phase_d_legacy_catalog_with_data_upgrades_and_reads() {
        use crate::{DuckLakeCatalog, DuckLakeTableWriter, SqliteMetadataProvider};
        use arrow::array::{Array, Int32Array, Int64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use datafusion::prelude::SessionContext;
        use object_store::local::LocalFileSystem;

        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("test.db");
        let data_path = temp.path().join("data");
        std::fs::create_dir_all(&data_path).unwrap();
        let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());

        // 1. Write real data (t(id int32) = [1,2,3]) with the current writer.
        {
            let writer = SqliteMetadataWriter::new_with_init(&conn_str)
                .await
                .unwrap();
            writer.set_data_path(data_path.to_str().unwrap()).unwrap();
            let store: std::sync::Arc<dyn object_store::ObjectStore> =
                std::sync::Arc::new(LocalFileSystem::new());
            let schema =
                std::sync::Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
            let batch = RecordBatch::try_new(
                schema,
                vec![std::sync::Arc::new(Int32Array::from(vec![1, 2, 3]))],
            )
            .unwrap();
            DuckLakeTableWriter::new(std::sync::Arc::new(writer), store)
                .unwrap()
                .write_table("main", "t", &[batch])
                .await
                .unwrap();
        }

        // 2. Downgrade ducklake_column to the LEGACY single-row-PK shape (what an
        //    older version wrote), preserving rows + column_ids. Data files untouched.
        {
            let pool = SqlitePool::connect(&conn_str).await.unwrap();
            sqlx::query("ALTER TABLE ducklake_column RENAME TO ducklake_column__tmp")
                .execute(&pool)
                .await
                .unwrap();
            sqlx::query(
                "CREATE TABLE ducklake_column (
                    column_id INTEGER PRIMARY KEY,
                    table_id INTEGER NOT NULL,
                    column_name VARCHAR NOT NULL,
                    column_type VARCHAR NOT NULL,
                    column_order INTEGER NOT NULL,
                    nulls_allowed BOOLEAN DEFAULT 1,
                    initial_default VARCHAR,
                    default_value VARCHAR,
                    parent_column INTEGER,
                    default_value_type VARCHAR,
                    default_value_dialect VARCHAR,
                    begin_snapshot INTEGER NOT NULL,
                    end_snapshot INTEGER
                )",
            )
            .execute(&pool)
            .await
            .unwrap();
            sqlx::query(
                "INSERT INTO ducklake_column
                     (column_id, table_id, column_name, column_type, column_order,
                      nulls_allowed, begin_snapshot, end_snapshot)
                 SELECT column_id, table_id, column_name, column_type, column_order,
                        nulls_allowed, begin_snapshot, end_snapshot
                 FROM ducklake_column__tmp",
            )
            .execute(&pool)
            .await
            .unwrap();
            sqlx::query("DROP TABLE ducklake_column__tmp")
                .execute(&pool)
                .await
                .unwrap();
            // Sanity: it really is the legacy PK shape now.
            let is_pk: i64 = sqlx::query_scalar(
                "SELECT pk FROM pragma_table_info('ducklake_column') WHERE name = 'column_id'",
            )
            .fetch_one(&pool)
            .await
            .unwrap();
            assert_eq!(is_pk, 1, "downgrade produced the legacy single-row PK");
            pool.close().await;
        }

        // 3. Re-open with the current version → initialize_schema runs the forward
        //    migration (legacy PK -> bare shape).
        let writer = SqliteMetadataWriter::new_with_init(&conn_str)
            .await
            .unwrap();
        let is_pk: i64 = sqlx::query_scalar(
            "SELECT pk FROM pragma_table_info('ducklake_column') WHERE name = 'column_id'",
        )
        .fetch_one(&writer.pool)
        .await
        .unwrap();
        assert_eq!(is_pk, 0, "re-open migrated the legacy PK away");

        // 4. The old data reads back intact through the full provider.
        let read_conn = format!("sqlite:{}", db_path.display());
        let provider = SqliteMetadataProvider::new(&read_conn).await.unwrap();
        let catalog = DuckLakeCatalog::new(provider).unwrap();
        let ctx = SessionContext::new();
        ctx.register_catalog("test", std::sync::Arc::new(catalog));
        let batches = ctx
            .sql("SELECT id FROM test.main.t ORDER BY id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let ids = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(
            ids.values(),
            &[1, 2, 3],
            "old-version data reads intact after upgrade"
        );

        // 5. The migrated catalog now supports promote (versioning enabled).
        let table_id: i64 =
            sqlx::query_scalar("SELECT table_id FROM ducklake_table WHERE table_name = 't'")
                .fetch_one(&writer.pool)
                .await
                .unwrap();
        writer.promote_column_type(table_id, "id", "int64").unwrap();
        let provider2 = SqliteMetadataProvider::new(&read_conn).await.unwrap();
        let catalog2 = DuckLakeCatalog::new(provider2).unwrap();
        let ctx2 = SessionContext::new();
        ctx2.register_catalog("test", std::sync::Arc::new(catalog2));
        let batches2 = ctx2
            .sql("SELECT id FROM test.main.t ORDER BY id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(batches2[0].schema().field(0).data_type(), &DataType::Int64);
        let ids2 = batches2[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(
            ids2.values(),
            &[1, 2, 3],
            "post-upgrade promote widens + reads intact"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_register_data_file() {
        let (writer, _temp) = create_test_writer().await;
        // Reserve (do not create) the snapshot: register_data_file inserts it.
        let snapshot_id = reserve_snapshot(&writer).await;
        let (schema_id, _) = writer
            .get_or_create_schema("main", None, snapshot_id)
            .unwrap();
        let (table_id, _) = writer
            .get_or_create_table(schema_id, "users", None, snapshot_id)
            .unwrap();

        let file = DataFileInfo::new("data.parquet", 1024, 100).with_footer_size(256);

        // register_data_file returns the committed snapshot id (head); the
        // first write commits snapshot 1.
        let committed = writer
            .register_data_file(
                table_id,
                "main",
                "users",
                snapshot_id,
                &file,
                WriteMode::Append,
                0,
                &[],
                &[],
            )
            .unwrap();
        assert_eq!(committed.snapshot_id, 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn snapshot_changes_are_atomic_with_append_and_replace() {
        let (writer, _temp) = create_test_writer().await;
        let snapshot_id = reserve_snapshot(&writer).await;
        let (schema_id, _) = writer
            .get_or_create_schema("main", None, snapshot_id)
            .unwrap();
        let (table_id, _) = writer
            .get_or_create_table(schema_id, "users", None, snapshot_id)
            .unwrap();

        let appended = writer
            .register_data_file(
                table_id,
                "main",
                "users",
                snapshot_id,
                &DataFileInfo::new("append.parquet", 1024, 100),
                WriteMode::Append,
                0,
                &[],
                &[],
            )
            .unwrap();

        let replacement_snapshot_id = reserve_snapshot(&writer).await;
        let commit_metadata = SnapshotCommitMetadata::new()
            .with_author("Jane Doe")
            .with_message("Replace imported data")
            .with_extra_info("opaque-application-value");
        let replaced = writer
            .register_data_file_with_commit_metadata(
                table_id,
                "main",
                "users",
                replacement_snapshot_id,
                &DataFileInfo::new("replace.parquet", 512, 50),
                WriteMode::Replace,
                appended.snapshot_id,
                &[],
                &[],
                &commit_metadata,
                None,
            )
            .unwrap();

        let rows = sqlx::query(
            "SELECT snapshot_id, changes_made, author, commit_message, commit_extra_info
             FROM ducklake_snapshot_changes
             ORDER BY snapshot_id",
        )
        .fetch_all(&writer.pool)
        .await
        .unwrap();
        let actual = rows
            .iter()
            .map(|row| {
                (
                    row.try_get::<i64, _>("snapshot_id").unwrap(),
                    row.try_get::<String, _>("changes_made").unwrap(),
                    row.try_get::<Option<String>, _>("author").unwrap(),
                    row.try_get::<Option<String>, _>("commit_message").unwrap(),
                    row.try_get::<Option<String>, _>("commit_extra_info")
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(
            actual,
            vec![
                (
                    appended.snapshot_id,
                    format!(
                        "created_schema:\"main\",created_table:\"main\".\"users\",\
                         inserted_into_table:{table_id}"
                    ),
                    None,
                    None,
                    None,
                ),
                (
                    replaced.snapshot_id,
                    format!("inserted_into_table:{table_id},deleted_from_table:{table_id}"),
                    Some("Jane Doe".to_string()),
                    Some("Replace imported data".to_string()),
                    Some("opaque-application-value".to_string()),
                ),
            ],
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn initial_replace_records_insertion_without_deletion() {
        let (writer, _temp) = create_test_writer().await;
        let snapshot_id = reserve_snapshot(&writer).await;
        let (schema_id, _) = writer
            .get_or_create_schema("main", None, snapshot_id)
            .unwrap();
        let (table_id, _) = writer
            .get_or_create_table(schema_id, "events", None, snapshot_id)
            .unwrap();

        let committed = writer
            .register_data_file(
                table_id,
                "main",
                "events",
                snapshot_id,
                &DataFileInfo::new("initial.parquet", 100, 3),
                WriteMode::Replace,
                0,
                &[],
                &[],
            )
            .unwrap();
        let changes_made: String = sqlx::query_scalar(
            "SELECT changes_made
             FROM ducklake_snapshot_changes
             WHERE snapshot_id = ?",
        )
        .bind(committed.snapshot_id)
        .fetch_one(&writer.pool)
        .await
        .unwrap();

        assert_eq!(
            changes_made,
            format!(
                "created_schema:\"main\",created_table:\"main\".\"events\",\
                 inserted_into_table:{table_id}"
            ),
        );
    }

    /// Reserve the next snapshot id the way `begin_write_transaction` now does
    /// (bump the monotonic counter WITHOUT inserting the row), so a subsequent
    /// `register_data_file` can insert it atomically at the commit. Tests that
    /// drive `register_data_file` directly must reserve (not `create_snapshot`)
    /// the snapshot they register, since registration owns the snapshot insert.
    async fn reserve_snapshot(writer: &SqliteMetadataWriter) -> i64 {
        sqlx::query("SELECT COALESCE(MAX(snapshot_id), 0) + 1 FROM ducklake_snapshot")
            .fetch_one(&writer.pool)
            .await
            .unwrap()
            .try_get(0)
            .unwrap()
    }

    /// Helper for the row-lineage tests: read back what `register_data_file`
    /// wrote into the catalog so we can assert on `row_id_start` and the
    /// stats counter directly.
    async fn read_row_id_start(writer: &SqliteMetadataWriter, path: &str) -> Option<i64> {
        let row = sqlx::query("SELECT row_id_start FROM ducklake_data_file WHERE path = ?")
            .bind(path)
            .fetch_one(&writer.pool)
            .await
            .unwrap();
        row.try_get(0).ok()
    }

    async fn read_table_stats(writer: &SqliteMetadataWriter, table_id: i64) -> (i64, i64, i64) {
        let row = sqlx::query(
            "SELECT record_count, next_row_id, file_size_bytes
             FROM ducklake_table_stats WHERE table_id = ?",
        )
        .bind(table_id)
        .fetch_one(&writer.pool)
        .await
        .unwrap();
        (
            row.try_get(0).unwrap(),
            row.try_get(1).unwrap(),
            row.try_get(2).unwrap(),
        )
    }

    #[allow(clippy::type_complexity)]
    async fn read_file_column_stats(
        writer: &SqliteMetadataWriter,
        path: &str,
    ) -> Vec<(
        i64,
        Option<String>,
        Option<String>,
        Option<i64>,
        Option<i64>,
    )> {
        sqlx::query(
            "SELECT s.column_id, s.min_value, s.max_value, s.null_count, s.value_count
             FROM ducklake_file_column_stats s
             JOIN ducklake_data_file d ON d.data_file_id = s.data_file_id
             WHERE d.path = ?
             ORDER BY s.column_id",
        )
        .bind(path)
        .fetch_all(&writer.pool)
        .await
        .unwrap()
        .into_iter()
        .map(|row| {
            (
                row.try_get(0).unwrap(),
                row.try_get(1).unwrap(),
                row.try_get(2).unwrap(),
                row.try_get(3).unwrap(),
                row.try_get(4).unwrap(),
            )
        })
        .collect()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn register_data_file_persists_column_stats() {
        // register_data_file must write one ducklake_file_column_stats row per
        // ColumnStat carried on the DataFileInfo, tied to the new data file's id.
        let (writer, _temp) = create_test_writer().await;
        let snap = reserve_snapshot(&writer).await;
        let (schema_id, _) = writer.get_or_create_schema("main", None, snap).unwrap();
        let (table_id, _) = writer
            .get_or_create_table(schema_id, "t", None, snap)
            .unwrap();

        let file = DataFileInfo::new("a.parquet", 100, 3).with_column_stats(vec![
            ColumnStat {
                column_id: 1,
                min_value: Some("1".to_string()),
                max_value: Some("3".to_string()),
                null_count: Some(0),
                value_count: Some(3),
                contains_nan: None,
                column_size_bytes: None,
            },
            ColumnStat {
                column_id: 2,
                min_value: Some("Alice".to_string()),
                max_value: Some("Bob".to_string()),
                null_count: Some(1),
                value_count: Some(2),
                contains_nan: None,
                column_size_bytes: None,
            },
        ]);
        writer
            .register_data_file(
                table_id,
                "main",
                "t",
                snap,
                &file,
                WriteMode::Append,
                0,
                &[],
                &[],
            )
            .unwrap();

        let rows = read_file_column_stats(&writer, "a.parquet").await;
        assert_eq!(
            rows,
            vec![
                (
                    1,
                    Some("1".to_string()),
                    Some("3".to_string()),
                    Some(0),
                    Some(3)
                ),
                (
                    2,
                    Some("Alice".to_string()),
                    Some("Bob".to_string()),
                    Some(1),
                    Some(2)
                ),
            ]
        );
    }

    async fn read_table_column_stats(
        writer: &SqliteMetadataWriter,
        table_id: i64,
    ) -> Vec<(i64, Option<bool>, Option<String>, Option<String>)> {
        sqlx::query(
            "SELECT column_id, contains_null, min_value, max_value
             FROM ducklake_table_column_stats WHERE table_id = ? ORDER BY column_id",
        )
        .bind(table_id)
        .fetch_all(&writer.pool)
        .await
        .unwrap()
        .into_iter()
        .map(|row| {
            (
                row.try_get(0).unwrap(),
                row.try_get(1).unwrap(),
                row.try_get(2).unwrap(),
                row.try_get(3).unwrap(),
            )
        })
        .collect()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn table_column_stats_widen_numerically_across_files() {
        // Two appends to a numeric column: global min/max must widen by numeric
        // value (min "9", max "10"), NOT lexically (which would rank "10" < "9").
        let (writer, _temp) = create_test_writer().await;
        let snap1 = reserve_snapshot(&writer).await;
        let (schema_id, _) = writer.get_or_create_schema("main", None, snap1).unwrap();
        let (table_id, _) = writer
            .get_or_create_table(schema_id, "t", None, snap1)
            .unwrap();
        let columns = vec![ColumnDef::new("n", "int32", true).unwrap()];
        let column_ids = vec![1_i64];

        let stat = |min: &str, max: &str, nulls: i64, vals: i64| {
            vec![ColumnStat {
                column_id: 1,
                min_value: Some(min.to_string()),
                max_value: Some(max.to_string()),
                null_count: Some(nulls),
                value_count: Some(vals),
                contains_nan: None,
                column_size_bytes: None,
            }]
        };

        writer
            .register_data_file(
                table_id,
                "main",
                "t",
                snap1,
                &DataFileInfo::new("a.parquet", 100, 3).with_column_stats(stat("9", "9", 0, 3)),
                WriteMode::Append,
                0,
                &columns,
                &column_ids,
            )
            .unwrap();
        let snap2 = reserve_snapshot(&writer).await;
        writer
            .register_data_file(
                table_id,
                "main",
                "t",
                snap2,
                &DataFileInfo::new("b.parquet", 100, 2).with_column_stats(stat("10", "10", 1, 1)),
                WriteMode::Append,
                0,
                &columns,
                &column_ids,
            )
            .unwrap();

        // Two files' per-file stats persisted.
        assert_eq!(read_file_column_stats(&writer, "a.parquet").await.len(), 1);
        assert_eq!(read_file_column_stats(&writer, "b.parquet").await.len(), 1);

        // Global roll-up widened numerically, and contains_null OR-reduced true.
        assert_eq!(
            read_table_column_stats(&writer, table_id).await,
            vec![(1, Some(true), Some("9".to_string()), Some("10".to_string()))]
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn table_column_stats_null_when_a_live_file_lacks_stats() {
        // File A has stats (null-free); file B is statless (empty column_stats).
        // Even though A reports contains_null=false, the roll-up must degrade to
        // NULL (unknown) — otherwise a reader treats the column as proven
        // null-free and drops genuine IS NULL rows. (Review finding M1.)
        let (writer, _temp) = create_test_writer().await;
        let snap1 = reserve_snapshot(&writer).await;
        let (schema_id, _) = writer.get_or_create_schema("main", None, snap1).unwrap();
        let (table_id, _) = writer
            .get_or_create_table(schema_id, "t", None, snap1)
            .unwrap();
        let columns = vec![ColumnDef::new("n", "int32", true).unwrap()];
        let column_ids = vec![1_i64];

        writer
            .register_data_file(
                table_id,
                "main",
                "t",
                snap1,
                &DataFileInfo::new("a.parquet", 100, 3).with_column_stats(vec![ColumnStat {
                    column_id: 1,
                    min_value: Some("1".to_string()),
                    max_value: Some("1".to_string()),
                    null_count: Some(0),
                    value_count: Some(3),
                    contains_nan: None,
                    column_size_bytes: None,
                }]),
                WriteMode::Append,
                0,
                &columns,
                &column_ids,
            )
            .unwrap();
        let snap2 = reserve_snapshot(&writer).await;
        writer
            .register_data_file(
                table_id,
                "main",
                "t",
                snap2,
                // Statless: no per-column stats attached.
                &DataFileInfo::new("b.parquet", 100, 2),
                WriteMode::Append,
                0,
                &columns,
                &column_ids,
            )
            .unwrap();

        assert_eq!(
            read_table_column_stats(&writer, table_id).await,
            Vec::new(),
            "a statless file leaves the column with no roll-up row (unknown)"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn row_id_start_advances_across_inserts() {
        // Two INSERTs against the same table should hand out non-overlapping
        // [row_id_start, row_id_start + record_count) ranges. Counter is
        // initialized lazily on first register_data_file.
        let (writer, _temp) = create_test_writer().await;
        // Two separate writes (each registers its own snapshot atomically).
        let snap1 = reserve_snapshot(&writer).await;
        let (schema_id, _) = writer.get_or_create_schema("main", None, snap1).unwrap();
        let (table_id, _) = writer
            .get_or_create_table(schema_id, "t", None, snap1)
            .unwrap();

        writer
            .register_data_file(
                table_id,
                "main",
                "t",
                snap1,
                &DataFileInfo::new("a.parquet", 100, 3),
                WriteMode::Append,
                0, // base_snapshot: unused for Append (no conflict check)
                &[],
                &[],
            )
            .unwrap();
        let snap2 = reserve_snapshot(&writer).await;
        writer
            .register_data_file(
                table_id,
                "main",
                "t",
                snap2,
                &DataFileInfo::new("b.parquet", 250, 7),
                WriteMode::Append,
                0, // base_snapshot: unused for Append (no conflict check)
                &[],
                &[],
            )
            .unwrap();

        assert_eq!(read_row_id_start(&writer, "a.parquet").await, Some(0));
        assert_eq!(read_row_id_start(&writer, "b.parquet").await, Some(3));

        let (records, next, bytes) = read_table_stats(&writer, table_id).await;
        assert_eq!(records, 10, "record_count = 3 + 7");
        assert_eq!(next, 10, "next_row_id advances by sum of record_counts");
        assert_eq!(bytes, 350, "file_size_bytes accumulates");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn end_table_files_preserves_next_row_id() {
        // Replace leaves ducklake_table_stats as official's DELETE does
        // (record_count is the gross count of rows ever inserted) and keeps
        // next_row_id monotonic so the next generation of files gets fresh,
        // non-overlapping rowids.
        let (writer, _temp) = create_test_writer().await;
        let snap1 = reserve_snapshot(&writer).await;
        let (schema_id, _) = writer.get_or_create_schema("main", None, snap1).unwrap();
        let (table_id, _) = writer
            .get_or_create_table(schema_id, "t", None, snap1)
            .unwrap();

        writer
            .register_data_file(
                table_id,
                "main",
                "t",
                snap1,
                &DataFileInfo::new("a.parquet", 100, 5),
                WriteMode::Append,
                0, // base_snapshot: unused for Append (no conflict check)
                &[],
                &[],
            )
            .unwrap();

        let snap2 = writer.create_snapshot().unwrap();
        writer.end_table_files(table_id, snap2).unwrap();

        let (records, next, bytes) = read_table_stats(&writer, table_id).await;
        assert_eq!(records, 5, "record_count stays the gross count");
        assert_eq!(next, 5, "next_row_id preserved (monotonic across lifetime)");
        assert!(bytes > 0, "file_size_bytes left as it was");

        let snap3 = reserve_snapshot(&writer).await;
        writer
            .register_data_file(
                table_id,
                "main",
                "t",
                snap3,
                &DataFileInfo::new("b.parquet", 200, 2),
                WriteMode::Append,
                0, // base_snapshot: unused for Append (no conflict check)
                &[],
                &[],
            )
            .unwrap();
        assert_eq!(
            read_row_id_start(&writer, "b.parquet").await,
            Some(5),
            "post-replace files must start at the preserved counter, not 0",
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn row_id_start_works_when_stats_row_missing() {
        // Defensive path: a table that existed before this writer started
        // maintaining ducklake_table_stats. First register_data_file must
        // self-initialize the stats row rather than fail.
        let (writer, _temp) = create_test_writer().await;
        let snapshot_id = reserve_snapshot(&writer).await;
        let (schema_id, _) = writer
            .get_or_create_schema("main", None, snapshot_id)
            .unwrap();
        let (table_id, _) = writer
            .get_or_create_table(schema_id, "legacy", None, snapshot_id)
            .unwrap();

        // Simulate a "legacy" table by deleting any stats row that
        // get_or_create_table may have written (it doesn't today, but be
        // explicit so the test stays meaningful if that changes).
        sqlx::query("DELETE FROM ducklake_table_stats WHERE table_id = ?")
            .bind(table_id)
            .execute(&writer.pool)
            .await
            .unwrap();

        writer
            .register_data_file(
                table_id,
                "main",
                "legacy",
                snapshot_id,
                &DataFileInfo::new("a.parquet", 50, 4),
                WriteMode::Append,
                0, // base_snapshot: unused for Append (no conflict check)
                &[],
                &[],
            )
            .unwrap();
        assert_eq!(read_row_id_start(&writer, "a.parquet").await, Some(0));
        let (records, next, _) = read_table_stats(&writer, table_id).await;
        assert_eq!(records, 4);
        assert_eq!(next, 4);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_end_table_files() {
        let (writer, _temp) = create_test_writer().await;
        let snapshot1 = reserve_snapshot(&writer).await;
        let (schema_id, _) = writer
            .get_or_create_schema("main", None, snapshot1)
            .unwrap();
        let (table_id, _) = writer
            .get_or_create_table(schema_id, "users", None, snapshot1)
            .unwrap();

        // Register a file
        let file = DataFileInfo::new("data1.parquet", 1024, 100);
        writer
            .register_data_file(
                table_id,
                "main",
                "users",
                snapshot1,
                &file,
                WriteMode::Append,
                0,
                &[],
                &[],
            )
            .unwrap();

        // End files at new snapshot
        let snapshot2 = writer.create_snapshot().unwrap();
        let ended = writer.end_table_files(table_id, snapshot2).unwrap();
        assert_eq!(ended, 1);

        // End again should affect 0 files
        let ended2 = writer.end_table_files(table_id, snapshot2).unwrap();
        assert_eq!(ended2, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn initialize_schema_migrates_directory_paths_once() {
        let (writer, _temp) = create_test_writer().await;
        writer.set_data_path("/tmp/ducklake-data").unwrap();
        let snapshot_id = writer.create_snapshot().unwrap();
        let (schema_id, _) = writer
            .get_or_create_schema("main", None, snapshot_id)
            .unwrap();
        writer
            .get_or_create_table(schema_id, "users", None, snapshot_id)
            .unwrap();
        async fn paths(pool: &SqlitePool) -> (String, String, String) {
            let data_path: String = sqlx::query_scalar(
                "SELECT value FROM ducklake_metadata WHERE key = 'data_path' AND scope IS NULL",
            )
            .fetch_one(pool)
            .await
            .unwrap();
            let schema_path: String = sqlx::query_scalar("SELECT path FROM ducklake_schema")
                .fetch_one(pool)
                .await
                .unwrap();
            let table_path: String = sqlx::query_scalar("SELECT path FROM ducklake_table")
                .fetch_one(pool)
                .await
                .unwrap();
            (data_path, schema_path, table_path)
        }
        assert_eq!(
            paths(&writer.pool).await,
            (
                "/tmp/ducklake-data/".to_string(),
                "main/".to_string(),
                "users/".to_string()
            )
        );

        sqlx::query(
            "UPDATE ducklake_metadata SET value = '/tmp/ducklake-data' WHERE key = 'data_path'",
        )
        .execute(&writer.pool)
        .await
        .unwrap();
        sqlx::query("UPDATE ducklake_schema SET path = 'main'")
            .execute(&writer.pool)
            .await
            .unwrap();
        sqlx::query("UPDATE ducklake_table SET path = 'users'")
            .execute(&writer.pool)
            .await
            .unwrap();
        for _ in 0..2 {
            writer.initialize_schema().unwrap();
            assert_eq!(
                paths(&writer.pool).await,
                (
                    "/tmp/ducklake-data/".to_string(),
                    "main/".to_string(),
                    "users/".to_string()
                )
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_data_path() {
        let (writer, _temp) = create_test_writer().await;

        // Set data path
        writer.set_data_path("/data/path").unwrap();

        // Get data path
        let path = writer.get_data_path().unwrap();
        assert_eq!(path, "/data/path/");

        // Update data path
        writer.set_data_path("/new/path").unwrap();
        let path2 = writer.get_data_path().unwrap();
        assert_eq!(path2, "/new/path/");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_or_create_schema_empty_name_rejected() {
        let (writer, _temp) = create_test_writer().await;
        let snapshot_id = writer.create_snapshot().unwrap();
        let result = writer.get_or_create_schema("", None, snapshot_id);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("empty"), "Expected 'empty' in: {err_msg}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_or_create_schema_control_char_rejected() {
        let (writer, _temp) = create_test_writer().await;
        let snapshot_id = writer.create_snapshot().unwrap();
        let result = writer.get_or_create_schema("bad\0schema", None, snapshot_id);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("control character"),
            "Expected 'control character' in: {err_msg}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_begin_write_transaction_empty_schema_name_rejected() {
        let (writer, _temp) = create_test_writer().await;
        let columns = vec![ColumnDef::new("id", "int64", false).unwrap()];
        let result = writer.begin_write_transaction("", "table", &columns, WriteMode::Replace);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("empty"), "Expected 'empty' in: {err_msg}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_begin_write_transaction_empty_table_name_rejected() {
        let (writer, _temp) = create_test_writer().await;
        let columns = vec![ColumnDef::new("id", "int64", false).unwrap()];
        let result = writer.begin_write_transaction("main", "", &columns, WriteMode::Replace);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("empty"), "Expected 'empty' in: {err_msg}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_begin_write_transaction_control_char_names_rejected() {
        let (writer, _temp) = create_test_writer().await;
        let columns = vec![ColumnDef::new("id", "int64", false).unwrap()];

        // Control char in schema name
        let result =
            writer.begin_write_transaction("bad\nschema", "table", &columns, WriteMode::Replace);
        assert!(result.is_err());

        // Control char in table name
        let result =
            writer.begin_write_transaction("main", "bad\ttable", &columns, WriteMode::Replace);
        assert!(result.is_err());
    }

    async fn read_end_snapshot(writer: &SqliteMetadataWriter, path: &str) -> Option<i64> {
        sqlx::query("SELECT end_snapshot FROM ducklake_data_file WHERE path = ?")
            .bind(path)
            .fetch_one(&writer.pool)
            .await
            .unwrap()
            .try_get(0)
            .unwrap()
    }

    async fn max_snapshot(writer: &SqliteMetadataWriter) -> i64 {
        sqlx::query_scalar("SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot")
            .fetch_one(&writer.pool)
            .await
            .unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn retire_appends_since_rolls_back_pure_append() {
        let (writer, _temp) = create_test_writer().await;
        // Base generation: append file A -> commits the published snapshot.
        let s0 = reserve_snapshot(&writer).await;
        let (schema_id, _) = writer.get_or_create_schema("main", None, s0).unwrap();
        let (table_id, _) = writer
            .get_or_create_table(schema_id, "t", None, s0)
            .unwrap();
        let file_a = DataFileInfo::new("a.parquet", 1000, 100).with_footer_size(64);
        let base = writer
            .register_data_file(
                table_id,
                "main",
                "t",
                s0,
                &file_a,
                WriteMode::Append,
                0,
                &[],
                &[],
            )
            .unwrap()
            .snapshot_id;

        // Append file B on top (the orphan): begin_snapshot > base.
        let s1 = reserve_snapshot(&writer).await;
        let file_b = DataFileInfo::new("b.parquet", 2000, 50).with_footer_size(64);
        writer
            .register_data_file(
                table_id,
                "main",
                "t",
                s1,
                &file_b,
                WriteMode::Append,
                base,
                &[],
                &[],
            )
            .unwrap();

        let (rc_before, nri_before, size_before) = read_table_stats(&writer, table_id).await;
        assert_eq!((rc_before, nri_before, size_before), (150, 150, 3000));

        // Roll back the append delta above base.
        let new_snap = writer.retire_appends_since(table_id, base).unwrap();
        assert!(new_snap.is_some(), "expected a compensating snapshot");

        // File A stays live; file B retired.
        assert!(read_end_snapshot(&writer, "a.parquet").await.is_none());
        assert!(read_end_snapshot(&writer, "b.parquet").await.is_some());

        // Stats left as official's DELETE leaves them; next_row_id preserved
        // (rowids never reused).
        let (rc_after, nri_after, size_after) = read_table_stats(&writer, table_id).await;
        assert_eq!(rc_after, 150, "record_count stays the gross count");
        assert_eq!(size_after, 3000, "file_size_bytes left as it was");
        assert_eq!(nri_after, 150, "next_row_id preserved");

        // Idempotent: nothing left above base -> no-op, no content-free snapshot.
        let head = max_snapshot(&writer).await;
        assert!(
            writer
                .retire_appends_since(table_id, base)
                .unwrap()
                .is_none()
        );
        assert_eq!(
            max_snapshot(&writer).await,
            head,
            "no snapshot minted on no-op"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn retire_appends_since_refuses_non_append_delta() {
        let (writer, _temp) = create_test_writer().await;
        let s0 = reserve_snapshot(&writer).await;
        let (schema_id, _) = writer.get_or_create_schema("main", None, s0).unwrap();
        let (table_id, _) = writer
            .get_or_create_table(schema_id, "t", None, s0)
            .unwrap();
        let file_a = DataFileInfo::new("a.parquet", 1000, 100).with_footer_size(64);
        let base = writer
            .register_data_file(
                table_id,
                "main",
                "t",
                s0,
                &file_a,
                WriteMode::Append,
                0,
                &[],
                &[],
            )
            .unwrap()
            .snapshot_id;

        // A non-append mutation after base: truncate ends file A
        // (begin_snapshot <= base, end_snapshot > base).
        writer.commit_truncate(table_id, "main", "t", base).unwrap();

        // The delta since base is no longer a pure append -> Conflict, so the
        // caller keeps its read freeze rather than exposing a half-reverted table.
        let err = writer.retire_appends_since(table_id, base).unwrap_err();
        assert!(
            matches!(err, crate::DuckLakeError::Conflict(_)),
            "expected Conflict, got {err:?}"
        );
    }
}
