//! DuckDB implementation of [`MetadataWriter`].
//!
//! A byte-compatible sibling of [`crate::metadata_writer_sqlite::SqliteMetadataWriter`]:
//! it writes the *exact same* DuckLake catalog tables (same names, same columns) so
//! that DuckDB's own `ducklake` extension — and this crate's
//! [`crate::DuckdbMetadataProvider`] — can read back what it writes.
//!
//! Scope is the legacy, single-catalog layout selected by `write-duckdb`.
//! Inserts, replacements, positional deletes, updates, compaction, truncate,
//! append restoration, type promotion, and snapshot expiry share the SQLite
//! writer's transaction and conflict semantics. Multicatalog remains specific
//! to PostgreSQL, and [`MetadataWriter::catalog_id`] returns `None`.
//!
//! ## How this differs from the SQLite writer
//!
//! 1. **Synchronous driver.** The `duckdb` crate is rusqlite-like and sync, not
//!    sqlx/async, so there is no `block_on`: the trait's sync `fn`s run directly
//!    against a `duckdb::Connection`. A single connection guarded by a `Mutex`
//!    makes the writer `Send + Sync` and — since DuckLake file catalogs are
//!    single-writer — serializes commits, which is exactly the "write lock" the
//!    SQLite writer leans on. Each mutating method opens one DuckDB transaction
//!    (`BEGIN`/`COMMIT`, auto-rollback on drop) for atomicity.
//! 2. **Id allocation.** SQLite relies on `INTEGER PRIMARY KEY` rowid
//!    autoincrement for `schema_id` / `table_id` / `data_file_id` /
//!    `delete_file_id`; a plain DuckDB PK does *not* autoincrement, so those
//!    columns default from DuckDB **sequences** (`nextval(...)`). `snapshot_id`
//!    keeps SQLite's commit-ordered `MAX(snapshot_id)+1` allocation (assigned at
//!    the commit, never reserved at begin) and `column_id` keeps SQLite's
//!    monotonic counter row in `ducklake_metadata` bumped with `UPDATE ...
//!    RETURNING`. The column *layouts* are otherwise identical to the SQLite
//!    DDL; only the id-default mechanism and the SQL type spellings
//!    (BIGINT/VARCHAR/BOOLEAN/TIMESTAMP) change.
//! 3. **Ordering + conflict detection preserved.** The snapshot is allocated
//!    before the table reads, and a `Replace` commit aborts with
//!    [`crate::DuckLakeError::Conflict`] if any data file of the table has a
//!    `begin_snapshot`/`end_snapshot` newer than the base observed at begin.

use crate::Result;
use crate::error::{TypeChangeOperation, TypeChangeWriteMode};
use crate::inlined_reclaim::{SUPERSEDED_INLINED_TABLES_SQL, dead_table_inlined_storage};
use crate::maintenance::{
    CleanupCriteria, ExpireCriteria, ExpiredSnapshot, ScheduledFile, format_sql_timestamp,
};
use crate::metadata_provider::InlinedSystemColumns;
use crate::metadata_provider_duckdb::{catalog_inlined_system_columns, inlined_table_layout};
use crate::metadata_writer::directory_path;
use crate::metadata_writer::{
    ColumnDef, ColumnStat, CommitIds, CompactionOutputFile, CompactionSourceFile, ConflictScope,
    DataFileInfo, DeleteFileEntry, DeleteFileInfo, ExistingCatalogColumn,
    INLINED_INDEX_COLUMNS_SETTING, InlinedFlushDeleteFile, InlinedFlushTable, InlinedRowRef,
    MetadataWriter, MultiTableCommit, SnapshotCommitMetadata, SourceRetirement, StagedTableData,
    StagedTableWrite, WriteMode, WriteSetupResult, assign_column_ids, carried_row_id_start,
    catalog_column_defs, catalog_column_type_equal, catalog_column_type_requires_migration,
    catalog_columns_differ, encode_inlined_index_columns, inlined_delete_groups,
    inlined_flush_changes, inlined_flush_conflict, live_inlined_index_columns,
    parse_inlined_index_columns, quote_snapshot_name, quote_snapshot_table, snapshot_has_change,
    staged_table_write_changes, table_storage_changes, top_level_column_ids,
    validate_delete_entries, validate_inlined_flush, validate_inlined_index_columns, validate_name,
    validate_table_setting,
};
use crate::metadata_writer::{
    DroppedDataFile, InlinedFileDelete, InlinedFlushFileDelete, SQL_GLOBAL_INLINING_COLUMNS,
    SQL_TABLE_INLINING_COLUMNS, SourceFile, TableChanges, enables_inlining,
    inlined_deletion_conflict, reject_reserved_inlining_columns, rewritten_delete_conflict,
    rewritten_deletes_snapshot, row_mutation_source_files, source_file_id_list, staged_conflict,
    table_conflict, validate_row_mutation_stage,
};
use crate::partition::PartitionTransform;
use arrow::array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, Decimal128Array, FixedSizeBinaryArray,
    FixedSizeListArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
    IntervalMonthDayNanoArray, LargeBinaryArray, LargeListArray, LargeStringArray, ListArray,
    MapArray, StringArray, StructArray, Time64MicrosecondArray, UInt8Array, UInt16Array,
    UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, TimeUnit as ArrowTimeUnit};
use arrow::record_batch::RecordBatch;
use duckdb::types::{Decimal, TimeUnit as DuckdbTimeUnit, Value};
use duckdb::{Connection, OptionalExt, Transaction, params, params_from_iter};
use std::fs::OpenOptions;
use std::sync::{Arc, Mutex, MutexGuard};

fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

fn inlined_duckdb_value(array: &dyn Array, row: usize) -> Result<Value> {
    if array.is_null(row) {
        return Ok(Value::Null);
    }
    if matches!(
        array.data_type(),
        DataType::Date32 | DataType::Timestamp(_, _)
    ) {
        // DuckDB's text, which the statement casts to the column's type:
        // duckdb-rs binds dates and timestamps through a microsecond
        // conversion that overflows on, or refuses, `infinity`, `-infinity`
        // and the far ends of the range. A value DuckDB's type does not hold
        // is refused, as the text backends refuse it.
        return Ok(Value::Text(crate::metadata_writer::inlined_text_value(
            array, row,
        )?));
    }
    if matches!(array.data_type(), DataType::Time64(_)) {
        crate::metadata_writer::inlined_text_value(array, row)?;
    }
    macro_rules! value {
        ($array:ty, $variant:ident) => {
            Value::$variant(
                array
                    .as_any()
                    .downcast_ref::<$array>()
                    .expect("Arrow data type and array implementation agree")
                    .value(row),
            )
        };
    }
    Ok(match array.data_type() {
        DataType::Boolean => value!(BooleanArray, Boolean),
        DataType::Int8 => value!(Int8Array, TinyInt),
        DataType::Int16 => value!(Int16Array, SmallInt),
        DataType::Int32 => value!(Int32Array, Int),
        DataType::Int64 => value!(Int64Array, BigInt),
        DataType::UInt8 => value!(UInt8Array, UTinyInt),
        DataType::UInt16 => value!(UInt16Array, USmallInt),
        DataType::UInt32 => value!(UInt32Array, UInt),
        DataType::UInt64 => value!(UInt64Array, UBigInt),
        DataType::Float32 => value!(Float32Array, Float),
        DataType::Float64 => value!(Float64Array, Double),
        DataType::Decimal128(precision, scale) => Value::Decimal(
            Decimal::new(
                *precision,
                u8::try_from(*scale).map_err(|e| {
                    crate::DuckLakeError::Unsupported(format!(
                        "DuckDB cannot inline decimal scale {scale}: {e}",
                    ))
                })?,
                array
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .expect("Arrow data type and array implementation agree")
                    .value(row),
            )
            .map_err(|e| {
                crate::DuckLakeError::Unsupported(format!(
                    "DuckDB cannot inline decimal({precision}, {scale}): {e}",
                ))
            })?,
        ),
        DataType::Time64(ArrowTimeUnit::Microsecond) => Value::Time64(
            DuckdbTimeUnit::Microsecond,
            array
                .as_any()
                .downcast_ref::<Time64MicrosecondArray>()
                .expect("Arrow data type and array implementation agree")
                .value(row),
        ),
        DataType::Interval(arrow::datatypes::IntervalUnit::MonthDayNano) => {
            let value = array
                .as_any()
                .downcast_ref::<IntervalMonthDayNanoArray>()
                .expect("Arrow data type and array implementation agree")
                .value(row);
            if value.nanoseconds % 1_000 != 0 {
                return Err(crate::DuckLakeError::Unsupported(format!(
                    "DuckDB INTERVAL cannot inline sub-microsecond nanoseconds: {}",
                    value.nanoseconds
                )));
            }
            Value::Interval {
                months: value.months,
                days: value.days,
                nanos: value.nanoseconds,
            }
        },
        DataType::Utf8 => Value::Text(
            array
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("Arrow data type and array implementation agree")
                .value(row)
                .to_owned(),
        ),
        DataType::LargeUtf8 => Value::Text(
            array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("Arrow data type and array implementation agree")
                .value(row)
                .to_owned(),
        ),
        DataType::Binary => Value::Blob(
            array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .expect("Arrow data type and array implementation agree")
                .value(row)
                .to_vec(),
        ),
        DataType::LargeBinary => Value::Blob(
            array
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .expect("Arrow data type and array implementation agree")
                .value(row)
                .to_vec(),
        ),
        DataType::BinaryView => Value::Blob(
            array
                .as_any()
                .downcast_ref::<BinaryViewArray>()
                .expect("Arrow data type and array implementation agree")
                .value(row)
                .to_vec(),
        ),
        DataType::FixedSizeBinary(size) if *size != 16 => Value::Blob(
            array
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .expect("Arrow data type and array implementation agree")
                .value(row)
                .to_vec(),
        ),
        _ => Value::Text(crate::metadata_writer::inlined_text_value(array, row)?),
    })
}

// Scalar leaves use backend-native DDL aliases; nested values use recursive native DDL and bind
// only their scalar leaves.
fn duckdb_type_inlines(data_type: &DataType) -> bool {
    crate::nested_inline::type_supports_inlining(data_type)
}

/// The DuckDB type of an inlined column of `data_type`, declared as DuckLake
/// type `ducklake_type`.
///
/// Built only from this crate's own type mapping, never from the stored type
/// text, which is spliced into DDL and a `CAST`: `ducklake_type` must parse as
/// a DuckLake type ([`crate::types::ducklake_to_arrow_type`] rejects unknown
/// names and trailing text) and only selects among fixed spellings where the
/// Arrow type alone does not say which DuckDB type official declares (`JSON`,
/// `UUID`, `TIMETZ`, the geometry types).
fn inlined_duckdb_type(data_type: &DataType, ducklake_type: &str) -> Result<String> {
    let unsupported = || {
        crate::DuckLakeError::UnsupportedType(format!(
            "no DuckDB type for an inlined column of DuckLake type '{ducklake_type}' ({data_type})"
        ))
    };
    if !ducklake_type.is_empty() {
        crate::types::ducklake_to_arrow_type(ducklake_type)?;
    }
    let declared = match ducklake_type.trim().to_ascii_lowercase().as_str() {
        "json" => Some("JSON"),
        "uuid" => Some("UUID"),
        "timetz" | "time with time zone" => Some("TIMETZ"),
        "geometry" => Some("GEOMETRY"),
        "point" => Some("POINT"),
        "linestring" => Some("LINESTRING"),
        "polygon" => Some("POLYGON"),
        "multipoint" => Some("MULTIPOINT"),
        "multilinestring" => Some("MULTILINESTRING"),
        "multipolygon" => Some("MULTIPOLYGON"),
        "geometrycollection" => Some("GEOMETRYCOLLECTION"),
        "linestring z" => Some("LINESTRING Z"),
        _ => None,
    };
    if let Some(declared) = declared {
        return Ok(declared.to_string());
    }
    crate::nested_inline::duckdb_type_name(data_type).ok_or_else(unsupported)
}

fn inlined_duckdb_value_sql(data_type: &DataType, ducklake_type: &str) -> Result<String> {
    Ok(format!(
        "CAST(? AS {})",
        inlined_duckdb_type(data_type, ducklake_type)?
    ))
}

struct EncodedDuckdbValue {
    sql: String,
    binds: Vec<Value>,
}

fn encode_duckdb_value(
    array: &dyn Array,
    row: usize,
    ducklake_type: &str,
    depth: usize,
) -> Result<EncodedDuckdbValue> {
    if depth > crate::types::MAX_NESTED_TYPE_DEPTH {
        return Err(crate::DuckLakeError::UnsupportedType(format!(
            "Nested inline value exceeds maximum depth {}",
            crate::types::MAX_NESTED_TYPE_DEPTH
        )));
    }

    let data_type = array.data_type();
    if let Some(nested_type) = crate::nested_inline::duckdb_type_name(data_type)
        && array.is_null(row)
        && !crate::nested_inline::scalar_type_supports_inlining(data_type)
    {
        return Ok(EncodedDuckdbValue {
            sql: format!("CAST(NULL AS {nested_type})"),
            binds: Vec::new(),
        });
    }

    match data_type {
        DataType::List(_) => {
            let array = array
                .as_any()
                .downcast_ref::<ListArray>()
                .expect("Arrow data type and array implementation agree");
            let offsets = array.value_offsets();
            encode_duckdb_list(
                array.values().as_ref(),
                offsets[row] as usize,
                offsets[row + 1] as usize,
                data_type,
                depth,
            )
        },
        DataType::LargeList(_) => {
            let array = array
                .as_any()
                .downcast_ref::<LargeListArray>()
                .expect("Arrow data type and array implementation agree");
            let offsets = array.value_offsets();
            encode_duckdb_list(
                array.values().as_ref(),
                offsets[row] as usize,
                offsets[row + 1] as usize,
                data_type,
                depth,
            )
        },
        DataType::FixedSizeList(_, size) => {
            let array = array
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .expect("Arrow data type and array implementation agree");
            let size = usize::try_from(*size).map_err(|e| {
                crate::DuckLakeError::UnsupportedType(format!(
                    "Invalid fixed-size list length {size}: {e}"
                ))
            })?;
            let start = row * size;
            encode_duckdb_list(
                array.values().as_ref(),
                start,
                start + size,
                data_type,
                depth,
            )
        },
        DataType::Struct(fields) => {
            let array = array
                .as_any()
                .downcast_ref::<StructArray>()
                .expect("Arrow data type and array implementation agree");
            let values = fields
                .iter()
                .enumerate()
                .map(|(index, field)| {
                    let value =
                        encode_duckdb_value(array.column(index).as_ref(), row, "", depth + 1)?;
                    Ok((
                        format!("{} := {}", quote_ident(field.name()), value.sql),
                        value.binds,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            let mut binds = Vec::new();
            let sql = values
                .into_iter()
                .map(|(sql, values)| {
                    binds.extend(values);
                    sql
                })
                .collect::<Vec<_>>()
                .join(", ");
            Ok(EncodedDuckdbValue {
                sql: format!(
                    "CAST(struct_pack({sql}) AS {})",
                    crate::nested_inline::duckdb_type_name(data_type)
                        .expect("supported struct has DuckDB DDL")
                ),
                binds,
            })
        },
        DataType::Map(_, _) => {
            let array = array
                .as_any()
                .downcast_ref::<MapArray>()
                .expect("Arrow data type and array implementation agree");
            let offsets = array.value_offsets();
            let start = offsets[row] as usize;
            let end = offsets[row + 1] as usize;
            let mut key_sql = Vec::with_capacity(end - start);
            let mut value_sql = Vec::with_capacity(end - start);
            let mut key_binds = Vec::new();
            let mut value_binds = Vec::new();
            for index in start..end {
                let key = encode_duckdb_value(array.keys().as_ref(), index, "", depth + 1)?;
                key_sql.push(key.sql);
                key_binds.extend(key.binds);
                let value = encode_duckdb_value(array.values().as_ref(), index, "", depth + 1)?;
                value_sql.push(value.sql);
                value_binds.extend(value.binds);
            }
            key_binds.extend(value_binds);
            Ok(EncodedDuckdbValue {
                sql: format!(
                    "CAST(map([{}], [{}]) AS {})",
                    key_sql.join(", "),
                    value_sql.join(", "),
                    crate::nested_inline::duckdb_type_name(data_type)
                        .expect("supported map has DuckDB DDL")
                ),
                binds: key_binds,
            })
        },
        _ => Ok(EncodedDuckdbValue {
            sql: inlined_duckdb_value_sql(data_type, ducklake_type)?,
            binds: vec![inlined_duckdb_value(array, row)?],
        }),
    }
}

fn encode_duckdb_list(
    values: &dyn Array,
    start: usize,
    end: usize,
    data_type: &DataType,
    depth: usize,
) -> Result<EncodedDuckdbValue> {
    let mut sql = Vec::with_capacity(end - start);
    let mut binds = Vec::new();
    for index in start..end {
        let value = encode_duckdb_value(values, index, "", depth + 1)?;
        sql.push(value.sql);
        binds.extend(value.binds);
    }
    Ok(EncodedDuckdbValue {
        sql: format!(
            "CAST([{}] AS {})",
            sql.join(", "),
            crate::nested_inline::duckdb_type_name(data_type)
                .expect("supported list has DuckDB DDL")
        ),
        binds,
    })
}

fn duckdb_inlined_index_columns(tx: &Transaction<'_>, table_id: i64) -> Result<Vec<String>> {
    let value: Option<String> = tx
        .query_row(
            "SELECT value FROM ducklake_metadata
             WHERE key = ? AND scope = 'table' AND scope_id = ?",
            params![INLINED_INDEX_COLUMNS_SETTING, table_id],
            |row| row.get(0),
        )
        .optional()?;
    let declared = parse_inlined_index_columns(value.as_deref().unwrap_or_default())?;
    let mut statement = tx.prepare(
        "SELECT column_name FROM ducklake_column
         WHERE table_id = ? AND end_snapshot IS NULL AND parent_column IS NULL",
    )?;
    let available = statement
        .query_map(params![table_id], |row| row.get::<_, String>(0))?
        .collect::<std::result::Result<std::collections::HashSet<_>, _>>()?;
    drop(statement);
    Ok(live_inlined_index_columns(declared, &available))
}

fn ensure_duckdb_physical_indexes(
    tx: &Transaction<'_>,
    table_id: i64,
    physical_name: &str,
) -> Result<()> {
    let declared = duckdb_inlined_index_columns(tx, table_id)?;
    let mut statement = tx.prepare("SELECT name FROM pragma_table_info(?)")?;
    let present = statement
        .query_map(params![physical_name], |row| row.get::<_, String>(0))?
        .collect::<std::result::Result<std::collections::HashSet<_>, _>>()?;
    drop(statement);
    for column in declared.iter().map(String::as_str) {
        if !present.contains(column) {
            continue;
        }
        let index_name = format!("{physical_name}_{column}_idx");
        tx.execute(
            &format!(
                "CREATE INDEX IF NOT EXISTS {} ON {}({})",
                quote_ident(&index_name),
                quote_ident(physical_name),
                quote_ident(column),
            ),
            [],
        )?;
    }
    Ok(())
}

fn id_list(ids: &[i64]) -> String {
    ids.iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

const RESOLVED_PATH: &str = "CASE
    WHEN NOT df.path_is_relative THEN df.path
    WHEN NOT t.path_is_relative THEN RTRIM(t.path, '/') || '/' || LTRIM(df.path, '/')
    ELSE RTRIM(s.path, '/') || '/' || RTRIM(t.path, '/') || '/' || LTRIM(df.path, '/')
END";

const REL_FLAG: &str =
    "(CASE WHEN df.path_is_relative AND t.path_is_relative AND s.path_is_relative
           THEN true ELSE false END)";

/// DuckLake catalog DDL for DuckDB.
///
/// Column-for-column identical to `SQL_CREATE_SCHEMA` in the SQLite writer; the
/// only changes are DuckDB spellings (BIGINT/VARCHAR/BOOLEAN/TIMESTAMP,
/// `DEFAULT true`) and the id-default mechanism. The `*_id_seq` sequences supply
/// the autoincrement that a plain DuckDB `PRIMARY KEY` does not; they are created
/// *before* the tables that reference them via `DEFAULT nextval(...)`. DuckDB
/// persists a sequence's value in the database file, so re-opening a catalog this
/// writer created continues handing out fresh, non-overlapping ids.
const SQL_CREATE_SCHEMA: &str = r#"
CREATE SEQUENCE IF NOT EXISTS ducklake_schema_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS ducklake_table_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS ducklake_data_file_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS ducklake_delete_file_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS ducklake_partition_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS ducklake_sort_id_seq START 1;

CREATE TABLE IF NOT EXISTS ducklake_metadata (
    key VARCHAR NOT NULL,
    value VARCHAR NOT NULL,
    scope VARCHAR,
    scope_id BIGINT
);

CREATE TABLE IF NOT EXISTS ducklake_snapshot (
    snapshot_id BIGINT PRIMARY KEY,
    snapshot_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    schema_version BIGINT NOT NULL DEFAULT 0,
    next_catalog_id BIGINT NOT NULL DEFAULT 1,
    next_file_id BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ducklake_snapshot_changes (
    snapshot_id BIGINT PRIMARY KEY,
    changes_made VARCHAR,
    author VARCHAR,
    commit_message VARCHAR,
    commit_extra_info VARCHAR
);

CREATE TABLE IF NOT EXISTS ducklake_schema_versions (
    begin_snapshot BIGINT NOT NULL,
    schema_version BIGINT NOT NULL,
    table_id BIGINT NOT NULL,
    UNIQUE (table_id, begin_snapshot)
);

CREATE TABLE IF NOT EXISTS ducklake_schema (
    schema_id BIGINT PRIMARY KEY DEFAULT nextval('ducklake_schema_id_seq'),
    schema_uuid UUID DEFAULT uuid(),
    schema_name VARCHAR NOT NULL,
    path VARCHAR NOT NULL DEFAULT '',
    path_is_relative BOOLEAN NOT NULL DEFAULT true,
    begin_snapshot BIGINT NOT NULL,
    end_snapshot BIGINT
);

CREATE TABLE IF NOT EXISTS ducklake_table (
    table_id BIGINT PRIMARY KEY DEFAULT nextval('ducklake_table_id_seq'),
    table_uuid UUID DEFAULT uuid(),
    schema_id BIGINT NOT NULL,
    table_name VARCHAR NOT NULL,
    path VARCHAR NOT NULL DEFAULT '',
    path_is_relative BOOLEAN NOT NULL DEFAULT true,
    begin_snapshot BIGINT NOT NULL,
    end_snapshot BIGINT
);

-- Bare table (no PK, no NOT NULL), upstream's exact column order, so a column
-- can be versioned by [begin_snapshot, end_snapshot) and a type promotion can
-- write a second row sharing the same column_id. Mirrors the SQLite writer's
-- `ducklake_column`. The four `*default*` columns + `parent_column` are left
-- NULL (no nested-type / column-default writes here).
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

CREATE TABLE IF NOT EXISTS ducklake_file_variant_stats (
    data_file_id BIGINT,
    table_id BIGINT,
    column_id BIGINT,
    variant_path VARCHAR,
    shredded_type VARCHAR,
    column_size_bytes BIGINT,
    value_count BIGINT,
    null_count BIGINT,
    min_value VARCHAR,
    max_value VARCHAR,
    contains_nan BOOLEAN,
    extra_stats VARCHAR
);

CREATE TABLE IF NOT EXISTS ducklake_column_mapping (
    mapping_id BIGINT,
    table_id BIGINT,
    type VARCHAR
);

CREATE TABLE IF NOT EXISTS ducklake_name_mapping (
    mapping_id BIGINT,
    column_id BIGINT,
    source_name VARCHAR,
    target_field_id BIGINT,
    parent_column BIGINT,
    is_partition BOOLEAN
);

CREATE TABLE IF NOT EXISTS ducklake_macro (
    schema_id BIGINT,
    macro_id BIGINT,
    macro_name VARCHAR,
    begin_snapshot BIGINT,
    end_snapshot BIGINT
);

CREATE TABLE IF NOT EXISTS ducklake_macro_impl (
    macro_id BIGINT,
    impl_id BIGINT,
    dialect VARCHAR,
    sql VARCHAR,
    type VARCHAR
);

CREATE TABLE IF NOT EXISTS ducklake_macro_parameters (
    macro_id BIGINT,
    impl_id BIGINT,
    column_id BIGINT,
    parameter_name VARCHAR,
    parameter_type VARCHAR,
    default_value VARCHAR,
    default_value_type VARCHAR
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

CREATE TABLE IF NOT EXISTS ducklake_data_file (
    data_file_id BIGINT PRIMARY KEY DEFAULT nextval('ducklake_data_file_id_seq'),
    table_id BIGINT NOT NULL,
    path VARCHAR NOT NULL,
    path_is_relative BOOLEAN NOT NULL DEFAULT true,
    file_size_bytes BIGINT NOT NULL,
    footer_size BIGINT,
    encryption_key VARCHAR,
    record_count BIGINT,
    row_id_start BIGINT,
    mapping_id BIGINT,
    begin_snapshot BIGINT NOT NULL,
    end_snapshot BIGINT,
    file_order BIGINT,
    file_format VARCHAR,
    partial_max BIGINT,
    -- References ducklake_partition_info.partition_id; NULL when unpartitioned.
    partition_id BIGINT
);

CREATE TABLE IF NOT EXISTS ducklake_snapshot_changes (
    snapshot_id BIGINT PRIMARY KEY,
    changes_made VARCHAR NOT NULL,
    author VARCHAR,
    commit_message VARCHAR,
    commit_extra_info VARCHAR
);

CREATE TABLE IF NOT EXISTS ducklake_table_stats (
    table_id BIGINT PRIMARY KEY,
    record_count BIGINT NOT NULL DEFAULT 0,
    next_row_id BIGINT NOT NULL DEFAULT 0,
    file_size_bytes BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ducklake_inlined_data_tables (
    table_id BIGINT NOT NULL,
    table_name VARCHAR NOT NULL,
    schema_version BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS ducklake_snapshot_changes (
    snapshot_id BIGINT PRIMARY KEY,
    changes_made VARCHAR NOT NULL,
    author VARCHAR,
    commit_message VARCHAR,
    commit_extra_info VARCHAR
);

-- Per-file, per-column zone maps (DuckLake spec) — powers file pruning.
-- Column set mirrors the official extension and the other backends.
CREATE TABLE IF NOT EXISTS ducklake_file_column_stats (
    data_file_id BIGINT NOT NULL,
    table_id BIGINT NOT NULL,
    column_id BIGINT NOT NULL,
    column_size_bytes BIGINT,
    value_count BIGINT,
    null_count BIGINT,
    min_value VARCHAR,
    max_value VARCHAR,
    contains_nan BOOLEAN,
    extra_stats VARCHAR
);

-- Table-wide per-column roll-up (DuckLake spec) — feeds the optimizer.
CREATE TABLE IF NOT EXISTS ducklake_table_column_stats (
    table_id BIGINT NOT NULL,
    column_id BIGINT NOT NULL,
    contains_null BOOLEAN,
    contains_nan BOOLEAN,
    min_value VARCHAR,
    max_value VARCHAR,
    extra_stats VARCHAR
);

CREATE TABLE IF NOT EXISTS ducklake_delete_file (
    delete_file_id BIGINT PRIMARY KEY DEFAULT nextval('ducklake_delete_file_id_seq'),
    data_file_id BIGINT NOT NULL,
    table_id BIGINT NOT NULL,
    path VARCHAR NOT NULL,
    path_is_relative BOOLEAN NOT NULL DEFAULT true,
    format VARCHAR,
    file_size_bytes BIGINT NOT NULL,
    footer_size BIGINT,
    encryption_key VARCHAR,
    delete_count BIGINT,
    begin_snapshot BIGINT NOT NULL,
    end_snapshot BIGINT,
    partial_max BIGINT
);

CREATE TABLE IF NOT EXISTS ducklake_files_scheduled_for_deletion (
    data_file_id BIGINT NOT NULL,
    path VARCHAR NOT NULL,
    path_is_relative BOOLEAN NOT NULL DEFAULT true,
    schedule_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Partition spec generations (DuckLake spec); end_snapshot NULL == active.
CREATE TABLE IF NOT EXISTS ducklake_partition_info (
    partition_id BIGINT NOT NULL,
    table_id BIGINT NOT NULL,
    begin_snapshot BIGINT NOT NULL,
    end_snapshot BIGINT
);

-- Partition-key columns for a spec (DuckLake spec), ordered by partition_key_index.
CREATE TABLE IF NOT EXISTS ducklake_partition_column (
    partition_id BIGINT NOT NULL,
    table_id BIGINT NOT NULL,
    partition_key_index BIGINT NOT NULL,
    column_id BIGINT NOT NULL,
    transform VARCHAR NOT NULL
);

-- Per-file partition values (DuckLake spec): the value every row in the file
-- shares for a partition key, DuckDB-canonical VARCHAR (NULL is legal).
CREATE TABLE IF NOT EXISTS ducklake_file_partition_value (
    data_file_id BIGINT NOT NULL,
    table_id BIGINT NOT NULL,
    partition_key_index BIGINT NOT NULL,
    partition_value VARCHAR
);

-- The partition pre-filter (`crate::stats_filter`) reads this table once per
-- listing, restricted to one table's rows for one partition key; unindexed that
-- is a scan of every table's partition values. Both columns are fixed-width
-- deliberately: `partition_value` is unbounded text.
CREATE INDEX IF NOT EXISTS idx_file_partition_value_table_key
    ON ducklake_file_partition_value (table_id, partition_key_index);

-- Sort spec generations (DuckLake spec); end_snapshot NULL == active. sort_id is
-- allocated from the next_sort_id counter (like partition_id).
CREATE TABLE IF NOT EXISTS ducklake_sort_info (
    sort_id BIGINT NOT NULL,
    table_id BIGINT NOT NULL,
    begin_snapshot BIGINT NOT NULL,
    end_snapshot BIGINT
);

-- Sort-key expressions for a spec (DuckLake spec), ordered by sort_key_index.
-- expression is a sort expression in `dialect` (this crate produces bare column
-- names under `duckdb`); sort_direction ASC/DESC; null_order NULLS_FIRST/LAST.
CREATE TABLE IF NOT EXISTS ducklake_sort_expression (
    sort_id BIGINT NOT NULL,
    table_id BIGINT NOT NULL,
    sort_key_index BIGINT NOT NULL,
    expression VARCHAR NOT NULL,
    dialect VARCHAR NOT NULL,
    sort_direction VARCHAR NOT NULL,
    null_order VARCHAR NOT NULL
);
"#;

/// DuckDB-based metadata writer for DuckLake catalogs.
///
/// Holds a single read-write `duckdb::Connection` behind a `Mutex` (cloneable via
/// `Arc`, sharing the one connection). See the module docs for the differences
/// from the SQLite writer.
#[derive(Debug, Clone)]
pub struct DuckdbMetadataWriter {
    conn: Arc<Mutex<Connection>>,
    catalog_path: String,
    commit_lock: Arc<Mutex<()>>,
}

impl DuckdbMetadataWriter {
    /// [`MetadataWriter::register_data_files_with_deletes`]: every file and
    /// every delete in one snapshot, each file drawing its own row ids.
    #[allow(clippy::too_many_arguments)]
    fn register_files_with_deletes(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        files: &[DataFileInfo],
        deletes: &[DeleteFileEntry],
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
    ) -> Result<CommitIds> {
        validate_delete_entries(mode, deletes)?;
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        let snapshot_id =
            finalize_snapshot(&tx, table_id, columns, column_ids, mode, base_snapshot)?;
        if !deletes.is_empty() {
            let ours = TableChanges {
                inserted_into: true,
                deleted_from: true,
                ..TableChanges::default()
            };
            let source = deleted_files_table(&tx, table_id, deletes)?;
            check_table_conflicts(&tx, source, base_snapshot, snapshot_id, ours)?;
        }
        let sources = SourceFile::of_entries(deletes, false);
        detect_new_inlined_deletes(&tx, table_id, base_snapshot, &sources)?;
        let live_partition_id: Option<i64> = tx.query_row(
            "SELECT (SELECT partition_id FROM ducklake_partition_info
                     WHERE table_id = ? AND end_snapshot IS NULL LIMIT 1)",
            params![table_id],
            |row| row.get(0),
        )?;
        for file in files {
            crate::metadata_writer::enforce_partition_fence(table_id, live_partition_id, file)?;
        }
        seed_stats_if_missing(&tx, table_id)?;
        for file in files {
            let next_row_id = tx.query_row(
                "SELECT next_row_id FROM ducklake_table_stats WHERE table_id = ?",
                params![table_id],
                |row| row.get::<_, i64>(0),
            )?;
            let row_ids = crate::metadata_writer::allocate_row_id_start(next_row_id, file);
            let data_file_id = reserve_file_ids(&tx, 1)?[0];
            tx.execute(
                "INSERT INTO ducklake_data_file
                     (data_file_id, table_id, path, path_is_relative, file_size_bytes,
                      footer_size, record_count, row_id_start, begin_snapshot)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params![
                    data_file_id,
                    table_id,
                    file.path.as_str(),
                    file.path_is_relative,
                    file.file_size_bytes,
                    file.footer_size,
                    file.record_count,
                    row_ids.stored,
                    snapshot_id,
                ],
            )?;
            insert_file_column_stats(&tx, table_id, data_file_id, &file.column_stats)?;
            insert_partition_metadata(&tx, table_id, data_file_id, file)?;
            tx.execute(
                "UPDATE ducklake_table_stats
                 SET next_row_id = next_row_id + ?, record_count = record_count + ?,
                     file_size_bytes = file_size_bytes + ?
                 WHERE table_id = ?",
                params![row_ids.advance, file.record_count, file.file_size_bytes, table_id],
            )?;
        }
        merge_file_table_column_stats(
            &tx,
            table_id,
            &crate::metadata_writer::catalog_fields(columns, column_ids)?,
            &crate::metadata_writer::file_column_stats(files),
        )?;
        for entry in deletes {
            apply_delete_entry(&tx, table_id, base_snapshot, snapshot_id, entry)?;
        }
        record_table_write_changes(
            &tx,
            snapshot_id,
            table_id,
            schema_name,
            table_name,
            mode,
            !deletes.is_empty(),
            false,
            &SnapshotCommitMetadata::default(),
        )?;
        let schema_id = tx.query_row(
            "SELECT schema_id FROM ducklake_table WHERE table_id = ?",
            params![table_id],
            |row| row.get(0),
        )?;
        tx.commit()?;
        Ok(CommitIds {
            snapshot_id,
            schema_id,
            table_id,
        })
    }

    /// Open (creating if absent) a DuckDB catalog file for writing.
    pub fn new(path: impl Into<String>) -> Result<Self> {
        let catalog_path = path.into();
        let conn = Connection::open(&catalog_path)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            catalog_path,
            commit_lock: Arc::new(Mutex::new(())),
        })
    }

    /// Open the catalog and initialize the DuckLake schema tables.
    pub fn new_with_init(path: impl Into<String>) -> Result<Self> {
        let writer = Self::new(path)?;
        writer.initialize_schema()?;
        Ok(writer)
    }

    /// Return a metadata provider sharing this writer's DuckDB connection.
    ///
    /// DuckDB permits one process-level client for a writable catalog file, so
    /// embedded catalogs retain provider and writer views over the same
    /// synchronized connection.
    #[must_use]
    pub fn metadata_provider(&self) -> crate::DuckdbMetadataProvider {
        crate::metadata_provider_duckdb::DuckdbMetadataProvider::from_shared_connection(
            Arc::clone(&self.conn),
            self.catalog_path.clone(),
        )
    }

    /// Lock the shared connection. Panics only if a previous holder panicked
    /// mid-write (poisoned mutex), which cannot leave partial SQL committed
    /// because every mutating method wraps its work in a transaction.
    fn connection(&self) -> MutexGuard<'_, Connection> {
        self.conn.lock().expect("DuckDB connection mutex poisoned")
    }

    /// Expire snapshots and remove catalog rows no longer reachable from a
    /// surviving snapshot. Physical files are queued for later cleanup. A fully
    /// expired table's inlined data and inlined delete tables are dropped with its
    /// registry rows in the same transaction; inlined rows no surviving snapshot
    /// can read stay in tables still live, as official keeps them.
    pub fn expire_snapshots(&self, criteria: ExpireCriteria) -> Result<Vec<ExpiredSnapshot>> {
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        let most_recent: Option<i64> = tx.query_row(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot",
            [],
            |row| row.get(0),
        )?;
        let Some(most_recent) = most_recent else {
            tx.commit()?;
            return Ok(Vec::new());
        };
        let candidates = match criteria {
            ExpireCriteria::Versions(versions) => {
                let versions = versions
                    .into_iter()
                    .filter(|snapshot| *snapshot != 0 && *snapshot != most_recent)
                    .collect::<Vec<_>>();
                if versions.is_empty() {
                    tx.commit()?;
                    return Ok(Vec::new());
                }
                let mut statement = tx.prepare(&format!(
                    "SELECT snapshot_id, CAST(snapshot_time AS VARCHAR)
                     FROM ducklake_snapshot WHERE snapshot_id IN ({}) ORDER BY snapshot_id",
                    id_list(&versions)
                ))?;
                statement
                    .query_map([], |row| {
                        Ok(ExpiredSnapshot {
                            snapshot_id: row.get(0)?,
                            snapshot_time: row.get(1)?,
                        })
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?
            },
            ExpireCriteria::OlderThan(cutoff) => {
                let mut statement = tx.prepare(
                    "SELECT snapshot_id, CAST(snapshot_time AS VARCHAR)
                     FROM ducklake_snapshot
                     WHERE snapshot_id != 0 AND snapshot_id != ?
                       AND snapshot_time < CAST(? AS TIMESTAMP)
                     ORDER BY snapshot_id",
                )?;
                statement
                    .query_map(params![most_recent, format_sql_timestamp(&cutoff)], |row| {
                        Ok(ExpiredSnapshot {
                            snapshot_id: row.get(0)?,
                            snapshot_time: row.get(1)?,
                        })
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?
            },
        };
        if candidates.is_empty() {
            tx.commit()?;
            return Ok(Vec::new());
        }
        let expire_ids = candidates
            .iter()
            .map(|snapshot| snapshot.snapshot_id)
            .collect::<Vec<_>>();
        tx.execute_batch(&format!(
            "DELETE FROM ducklake_snapshot WHERE snapshot_id IN ({})",
            id_list(&expire_ids)
        ))?;

        let dead_tables = {
            let mut statement = tx.prepare(
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
            )?;
            statement
                .query_map([], |row| row.get::<_, i64>(0))?
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        let dead_table_filter = if dead_tables.is_empty() {
            "false".to_string()
        } else {
            format!("df.table_id IN ({})", id_list(&dead_tables))
        };
        let dead_data_files = {
            let mut statement = tx.prepare(&format!(
                "SELECT df.data_file_id FROM ducklake_data_file df
                 WHERE ({dead_table_filter}) OR (df.end_snapshot IS NOT NULL AND NOT EXISTS (
                     SELECT 1 FROM ducklake_snapshot
                     WHERE snapshot_id >= df.begin_snapshot AND snapshot_id < df.end_snapshot))"
            ))?;
            statement
                .query_map([], |row| row.get::<_, i64>(0))?
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        if !dead_data_files.is_empty() {
            let ids = id_list(&dead_data_files);
            tx.execute_batch(&format!(
                "INSERT INTO ducklake_files_scheduled_for_deletion
                     (data_file_id, path, path_is_relative)
                 SELECT df.data_file_id, {RESOLVED_PATH}, {REL_FLAG}
                 FROM ducklake_data_file df
                 JOIN ducklake_table t ON t.table_id = df.table_id
                 JOIN ducklake_schema s ON s.schema_id = t.schema_id
                 WHERE df.data_file_id IN ({ids});
                 DELETE FROM ducklake_data_file WHERE data_file_id IN ({ids})"
            ))?;
        }
        let dead_data_filter = if dead_data_files.is_empty() {
            "false".to_string()
        } else {
            format!("df.data_file_id IN ({})", id_list(&dead_data_files))
        };
        let dead_delete_table_filter = if dead_tables.is_empty() {
            "false".to_string()
        } else {
            format!("df.table_id IN ({})", id_list(&dead_tables))
        };
        let dead_delete_files = {
            let mut statement = tx.prepare(&format!(
                "SELECT df.delete_file_id FROM ducklake_delete_file df
                 WHERE ({dead_data_filter}) OR ({dead_delete_table_filter})
                    OR (df.end_snapshot IS NOT NULL AND NOT EXISTS (
                        SELECT 1 FROM ducklake_snapshot
                        WHERE snapshot_id >= df.begin_snapshot AND snapshot_id < df.end_snapshot))"
            ))?;
            statement
                .query_map([], |row| row.get::<_, i64>(0))?
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        if !dead_delete_files.is_empty() {
            let ids = id_list(&dead_delete_files);
            tx.execute_batch(&format!(
                "INSERT INTO ducklake_files_scheduled_for_deletion
                     (data_file_id, path, path_is_relative)
                 SELECT df.delete_file_id, {RESOLVED_PATH}, {REL_FLAG}
                 FROM ducklake_delete_file df
                 JOIN ducklake_table t ON t.table_id = df.table_id
                 JOIN ducklake_schema s ON s.schema_id = t.schema_id
                 WHERE df.delete_file_id IN ({ids});
                 DELETE FROM ducklake_delete_file WHERE delete_file_id IN ({ids})"
            ))?;
        }
        if !dead_tables.is_empty() {
            let ids = id_list(&dead_tables);
            // The table's inlined storage goes with it: its registered inlined data
            // tables and its inlined delete table, listed before the registry rows are
            // deleted and dropped after them, as official does.
            // A catalog this crate's writer created before data inlining has no registry.
            let has_registry = self.metadata_provider().has_inlined_data_registry(&tx)?;
            let registered = if has_registry {
                let mut statement = tx.prepare(&format!(
                    "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id IN ({ids})"
                ))?;
                statement
                    .query_map([], |row| row.get::<_, String>(0))?
                    .collect::<std::result::Result<Vec<_>, _>>()?
            } else {
                Vec::new()
            };
            let inlined_storage = dead_table_inlined_storage(registered, &dead_tables)?;
            tx.execute_batch(&format!(
                "DELETE FROM ducklake_table WHERE table_id IN ({ids});
                 DELETE FROM ducklake_table_stats WHERE table_id IN ({ids});
                 DELETE FROM ducklake_column WHERE table_id IN ({ids});
                 DELETE FROM ducklake_schema_versions WHERE table_id IN ({ids})"
            ))?;
            if has_registry {
                tx.execute_batch(&format!(
                    "DELETE FROM ducklake_inlined_data_tables WHERE table_id IN ({ids})"
                ))?;
            }
            for table in inlined_storage {
                tx.execute_batch(&format!("DROP TABLE IF EXISTS {}", quote_ident(&table)))?;
            }
        }
        tx.execute_batch(
            "DELETE FROM ducklake_schema
             WHERE end_snapshot IS NOT NULL AND NOT EXISTS (
                 SELECT 1 FROM ducklake_snapshot
                 WHERE snapshot_id >= ducklake_schema.begin_snapshot
                   AND snapshot_id < ducklake_schema.end_snapshot)",
        )?;
        tx.commit()?;
        Ok(candidates)
    }
}

/// Atomically reserve `n` consecutive ids from a monotonic counter row in
/// `ducklake_metadata` (seeded by `initialize_schema`), returning the LAST id of
/// the block — the reserved ids are `last - n + 1 ..= last`. Mirrors the SQLite
/// writer's `reserve_ids`; used for `column_id`, which is reserved at begin and
/// baked into the staged parquet's `field_id` metadata. `value` is stored as
/// text and cast through BIGINT so the `UPDATE ... RETURNING` is exact.
fn reserve_ids(tx: &Transaction<'_>, key: &str, n: i64) -> Result<i64> {
    let last: i64 = tx.query_row(
        "UPDATE ducklake_metadata
         SET value = CAST(CAST(value AS BIGINT) + ? AS VARCHAR)
         WHERE key = ? AND scope IS NULL
         RETURNING CAST(value AS BIGINT)",
        params![n, key],
        |row| row.get(0),
    )?;
    Ok(last)
}

fn reserve_sequence_ids(tx: &Transaction<'_>, sequence: &str, n: i64) -> Result<Vec<i64>> {
    let sql = format!("SELECT nextval('{sequence}')");
    (0..n)
        .map(|_| tx.query_row(&sql, [], |row| row.get(0)).map_err(Into::into))
        .collect()
}

fn reserve_file_ids(tx: &Transaction<'_>, n: i64) -> Result<Vec<i64>> {
    reserve_sequence_ids(tx, "ducklake_data_file_id_seq", n)
}

fn reserve_delete_file_ids(tx: &Transaction<'_>, n: i64) -> Result<Vec<i64>> {
    reserve_sequence_ids(tx, "ducklake_delete_file_id_seq", n)
}

/// Register a delete file an inlined flush wrote for `data_file_id`.
/// Register each of `entries`, a delete file that records its positions'
/// snapshots, against its data file. One that replaces the file's live delete
/// file carries every position of it, so the replaced row is removed and its
/// object scheduled for deletion, as official's `DeleteOverwrittenDeleteFiles`
/// does. Fails with `conflict()` when a data file's `end_snapshot` is not the
/// entry's, or the delete file it replaces (a live file's live one, a retired
/// file's latest one) is not the entry's.
fn replace_delete_files(
    tx: &Transaction<'_>,
    table_id: i64,
    entries: &[InlinedFlushFileDelete],
    conflict: impl Fn() -> crate::DuckLakeError,
) -> Result<()> {
    for entry in entries {
        let target_end: Option<Option<i64>> = tx
            .query_row(
                "SELECT end_snapshot FROM ducklake_data_file WHERE data_file_id = ?",
                params![entry.data_file_id],
                |row| row.get(0),
            )
            .optional()?;
        // A live file's live delete file, or a retired file's latest one.
        let current: Option<i64> = tx
            .query_row(
                "SELECT delete_file_id FROM ducklake_delete_file
                 WHERE data_file_id = ? AND (end_snapshot IS NULL OR ?)
                 ORDER BY begin_snapshot DESC LIMIT 1",
                params![entry.data_file_id, entry.data_file_end.is_some()],
                |row| row.get(0),
            )
            .optional()?;
        if target_end != Some(entry.data_file_end) || current != entry.expected_prev_delete_file {
            return Err(conflict());
        }
        if let Some(previous) = entry.expected_prev_delete_file {
            tx.execute(
                &format!(
                    "INSERT INTO ducklake_files_scheduled_for_deletion
                         (data_file_id, path, path_is_relative)
                     SELECT df.delete_file_id, {RESOLVED_PATH}, {REL_FLAG}
                     FROM ducklake_delete_file df
                     JOIN ducklake_table t ON t.table_id = df.table_id
                     JOIN ducklake_schema s ON s.schema_id = t.schema_id
                     WHERE df.delete_file_id = ?"
                ),
                params![previous],
            )?;
            tx.execute(
                "DELETE FROM ducklake_delete_file WHERE delete_file_id = ?",
                params![previous],
            )?;
        }
        insert_flush_delete_file(tx, table_id, entry.data_file_id, &entry.delete)?;
    }
    Ok(())
}

fn insert_flush_delete_file(
    tx: &Transaction<'_>,
    table_id: i64,
    data_file_id: i64,
    delete: &InlinedFlushDeleteFile,
) -> Result<()> {
    let delete_file_id = reserve_delete_file_ids(tx, 1)?[0];
    tx.execute(
        "INSERT INTO ducklake_delete_file
             (delete_file_id, data_file_id, table_id, path, path_is_relative,
              file_size_bytes, footer_size, delete_count, begin_snapshot, partial_max,
              end_snapshot)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            delete_file_id,
            data_file_id,
            table_id,
            delete.delete.path.as_str(),
            delete.delete.path_is_relative,
            delete.delete.file_size_bytes,
            delete.delete.footer_size,
            delete.delete.delete_count,
            delete.begin_snapshot,
            delete.partial_max,
            delete.end_snapshot,
        ],
    )?;
    Ok(())
}

/// Insert the next `ducklake_snapshot` row in commit order, carrying
/// `schema_version` forward (the pure-data-write default), and return
/// `(snapshot_id, schema_version)`.
///
/// Mirrors the SQLite writer's `insert_snapshot` — `MAX(snapshot_id)+1` assigned
/// at the commit, never reserved at begin — but split into a read then an insert
/// (DuckDB's single-writer connection is held under the `Mutex`, so no
/// concurrent writer can slip between them; the two-statement form avoids relying
/// on `INSERT ... SELECT ... RETURNING`). A DDL commit follows this with
/// [`bump_schema_version`].
fn insert_snapshot(tx: &Transaction<'_>) -> Result<(i64, i64)> {
    let (snapshot_id, schema_version): (i64, i64) = tx.query_row(
        "SELECT COALESCE(MAX(snapshot_id), 0) + 1, COALESCE(MAX(schema_version), 0)
         FROM ducklake_snapshot",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    tx.execute(
        "INSERT INTO ducklake_snapshot (snapshot_id, snapshot_time, schema_version)
         VALUES (?, CURRENT_TIMESTAMP, ?)",
        params![snapshot_id, schema_version],
    )?;
    tx.execute(
        "INSERT INTO ducklake_snapshot_changes (snapshot_id, changes_made)
         VALUES (?, NULL)",
        params![snapshot_id],
    )?;
    Ok((snapshot_id, schema_version))
}

fn record_snapshot_changes(
    tx: &Transaction<'_>,
    snapshot_id: i64,
    changes_made: &str,
    commit_metadata: &SnapshotCommitMetadata,
) -> Result<()> {
    let changes_made = (!changes_made.is_empty()).then_some(changes_made);
    tx.execute(
        "UPDATE ducklake_snapshot_changes
         SET changes_made = CASE
                 WHEN changes_made IS NULL THEN ?
                 WHEN ? IS NULL THEN changes_made
                 ELSE changes_made || ',' || ?
             END,
             author = ?,
             commit_message = ?,
             commit_extra_info = ?
         WHERE snapshot_id = ?",
        params![
            changes_made,
            changes_made,
            changes_made,
            commit_metadata.author(),
            commit_metadata.message(),
            commit_metadata.extra_info(),
            snapshot_id,
        ],
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn record_table_write_changes(
    tx: &Transaction<'_>,
    snapshot_id: i64,
    table_id: i64,
    schema_name: &str,
    table_name: &str,
    mode: WriteMode,
    has_deletes: bool,
    inlined: bool,
    commit_metadata: &SnapshotCommitMetadata,
) -> Result<()> {
    let replaced_existing_data: bool = tx.query_row(
        "SELECT EXISTS(
            SELECT 1 FROM ducklake_data_file
            WHERE table_id = ? AND end_snapshot = ?
         )",
        params![table_id, snapshot_id],
        |row| row.get(0),
    )?;
    let mut replaced_inlined = false;
    if mode == WriteMode::Replace {
        // A Replace over inline-only prior data ends inline rows, not files.
        for (inlined_table, system) in inlined_tables(tx, table_id)? {
            let sql = format!(
                "SELECT EXISTS(SELECT 1 FROM {} WHERE {} = ?)",
                quote_ident(&inlined_table),
                system.end_snapshot,
            );
            if tx.query_row(&sql, params![snapshot_id], |row| row.get(0))? {
                replaced_inlined = true;
                break;
            }
        }
    }

    let write_changes = table_storage_changes(
        table_id,
        Some(inlined),
        has_deletes || (mode == WriteMode::Replace && replaced_existing_data),
        replaced_inlined,
    );
    record_table_changes(
        tx,
        snapshot_id,
        table_id,
        schema_name,
        table_name,
        &write_changes,
        commit_metadata,
    )
}

fn record_table_changes(
    tx: &Transaction<'_>,
    snapshot_id: i64,
    table_id: i64,
    schema_name: &str,
    table_name: &str,
    write_changes: &str,
    commit_metadata: &SnapshotCommitMetadata,
) -> Result<()> {
    let (schema_begin_snapshot, table_begin_snapshot): (i64, i64) = tx.query_row(
        "SELECT s.begin_snapshot, t.begin_snapshot
         FROM ducklake_table t
         JOIN ducklake_schema s ON s.schema_id = t.schema_id
         WHERE t.table_id = ?",
        params![table_id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    let altered: bool = tx.query_row(
        "SELECT EXISTS(
            SELECT 1 FROM ducklake_schema_versions
            WHERE table_id = ? AND begin_snapshot = ?
         )",
        params![table_id, snapshot_id],
        |row| row.get(0),
    )?;
    let mut changes = Vec::new();
    if schema_begin_snapshot == snapshot_id {
        let entry = format!("created_schema:{}", quote_snapshot_name(schema_name));
        let recorded: Option<String> = tx
            .query_row(
                "SELECT changes_made FROM ducklake_snapshot_changes WHERE snapshot_id = ?",
                params![snapshot_id],
                |row| row.get(0),
            )
            .optional()?
            .flatten();
        if !snapshot_has_change(recorded.as_deref().unwrap_or_default(), &entry) {
            changes.push(entry);
        }
    }
    if table_begin_snapshot == snapshot_id {
        changes.push(format!(
            "created_table:{}",
            quote_snapshot_table(schema_name, table_name)
        ));
    } else if altered {
        changes.push(format!("altered_table:{table_id}"));
    }
    changes.push(write_changes.to_string());
    record_snapshot_changes(tx, snapshot_id, &changes.join(","), commit_metadata)
}

/// Bump the per-catalog monotonic `schema_version` on a DDL snapshot to
/// `prev_max + 1` (max over the OTHER snapshots, so re-running is stable) and
/// return the new value. Mirrors the SQLite writer's `bump_schema_version`.
fn bump_schema_version(tx: &Transaction<'_>, snapshot_id: i64) -> Result<i64> {
    let prev_max: i64 = tx.query_row(
        "SELECT COALESCE(MAX(schema_version), 0) FROM ducklake_snapshot WHERE snapshot_id <> ?",
        params![snapshot_id],
        |row| row.get(0),
    )?;
    let new_version = prev_max + 1;
    tx.execute(
        "UPDATE ducklake_snapshot SET schema_version = ? WHERE snapshot_id = ?",
        params![new_version, snapshot_id],
    )?;
    Ok(new_version)
}

/// Record a `ducklake_schema_versions` ledger row for a DDL that leaves the table
/// live (create, column add/remove/reorder). Mirrors the SQLite writer.
fn record_schema_version(
    tx: &Transaction<'_>,
    snapshot_id: i64,
    schema_version: i64,
    table_id: i64,
) -> Result<()> {
    tx.execute(
        "INSERT INTO ducklake_schema_versions (begin_snapshot, schema_version, table_id)
         VALUES (?, ?, ?)",
        params![snapshot_id, schema_version, table_id],
    )?;
    Ok(())
}

/// Seed the `ducklake_table_stats` row for a brand-new table if it is missing.
///
/// Replaces the SQLite writer's `INSERT OR IGNORE` with an explicit
/// exists-check, avoiding any dependency on DuckDB upsert syntax while producing
/// the identical result (a single zeroed stats row per table).
fn seed_stats_if_missing(tx: &Transaction<'_>, table_id: i64) -> Result<()> {
    let exists: Option<i64> = tx
        .query_row(
            "SELECT 1 FROM ducklake_table_stats WHERE table_id = ?",
            params![table_id],
            |row| row.get(0),
        )
        .optional()?;
    if exists.is_none() {
        tx.execute(
            "INSERT INTO ducklake_table_stats
                 (table_id, record_count, next_row_id, file_size_bytes)
             VALUES (?, 0, 0, 0)",
            params![table_id],
        )?;
    }
    Ok(())
}

/// Optimistic-concurrency check for a `Replace` commit (mirrors the SQLite
/// writer). If any data file or delete file of the table has `begin_snapshot` or
/// `end_snapshot` newer than `base_snapshot` (the head observed when this write
/// began), or an inlined data or inlined delete row was committed after it,
/// another writer changed the table in the meantime, so this `Replace` aborts
/// with [`crate::DuckLakeError::Conflict`] rather than clobbering it.
/// An inlined-data flush ([`ConflictScope::InlinedFlush`]) does not count delete
/// files: official DuckLake commits a flush over a concurrent delete file.
fn detect_replace_conflict(
    tx: &Transaction<'_>,
    table_id: i64,
    base_snapshot: i64,
    scope: ConflictScope,
) -> Result<()> {
    let conflicts: i64 = tx.query_row(
        "SELECT (SELECT COUNT(*) FROM ducklake_data_file
                 WHERE table_id = ? AND (begin_snapshot > ? OR end_snapshot > ?))
              + (SELECT COUNT(*) FROM ducklake_delete_file
                 WHERE ? AND table_id = ? AND (begin_snapshot > ? OR end_snapshot > ?))",
        params![
            table_id,
            base_snapshot,
            base_snapshot,
            scope.counts_delete_files(),
            table_id,
            base_snapshot,
            base_snapshot
        ],
        |row| row.get(0),
    )?;
    if conflicts > 0 {
        return Err(crate::DuckLakeError::Conflict(format!(
            "Replace on table {table_id} conflicts with a concurrent write committed since \
             snapshot {base_snapshot}; aborting (retry the write against the new generation)"
        )));
    }
    for (table_name, system) in inlined_tables(tx, table_id)? {
        let conflicts: i64 = tx.query_row(
            &format!(
                "SELECT COUNT(*) FROM {} WHERE {} > ? OR {} > ?",
                quote_ident(&table_name),
                system.begin_snapshot,
                system.end_snapshot,
            ),
            params![base_snapshot, base_snapshot],
            |row| row.get(0),
        )?;
        if conflicts > 0 {
            return Err(crate::DuckLakeError::Conflict(format!(
                "Replace on table {table_id} conflicts with inlined data committed since \
                 snapshot {base_snapshot}; aborting"
            )));
        }
    }
    let inlined_deletes = crate::metadata_provider::inlined_delete_table_name(table_id)?;
    let exists: bool = tx.query_row(
        "SELECT COUNT(*) > 0 FROM information_schema.tables WHERE table_name = ?",
        params![inlined_deletes.as_str()],
        |row| row.get(0),
    )?;
    if exists {
        let conflicts: i64 = tx.query_row(
            &format!(
                "SELECT COUNT(*) FROM {} WHERE begin_snapshot > ?",
                quote_ident(&inlined_deletes)
            ),
            params![base_snapshot],
            |row| row.get(0),
        )?;
        if conflicts > 0 {
            return Err(crate::DuckLakeError::Conflict(format!(
                "Replace on table {table_id} conflicts with an inlined delete committed since \
                 snapshot {base_snapshot}; aborting"
            )));
        }
    }
    Ok(())
}

fn detect_new_inlined_deletes(
    tx: &Transaction<'_>,
    table_id: i64,
    base_snapshot: i64,
    sources: &[SourceFile<'_>],
) -> Result<()> {
    if sources.is_empty() {
        return Ok(());
    }

    let table = crate::metadata_provider::inlined_delete_table_name(table_id)?;
    let exists: bool = tx.query_row(
        "SELECT COUNT(*) > 0 FROM information_schema.tables WHERE table_name = ?",
        params![table.as_str()],
        |row| row.get(0),
    )?;
    if !exists {
        return Ok(());
    }

    let mut statement = tx.prepare(&format!(
        "SELECT newer.file_id, newer.row_id FROM {} newer
         WHERE newer.file_id IN ({})
           AND (newer.begin_snapshot IS NULL OR (
             newer.begin_snapshot > ?
             AND (newer.row_id IS NULL OR NOT EXISTS (
               SELECT 1 FROM {} prior
               WHERE prior.file_id = newer.file_id
                 AND prior.row_id = newer.row_id
                 AND prior.begin_snapshot <= ?
             ))
           ))",
        quote_ident(&table),
        source_file_id_list(sources),
        quote_ident(&table),
    ))?;
    let concurrent = statement
        .query_map(params![base_snapshot, base_snapshot], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, Option<i64>>(1)?))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    inlined_deletion_conflict(table_id, base_snapshot, sources, &concurrent)
}

/// Retire the prior generation's still-visible data files and inlined rows at
/// `snapshot_id`. The `begin_snapshot < snapshot_id` guard spares files
/// registered for *this* snapshot. `ducklake_table_stats` is left as it is, as
/// official's `DELETE` leaves it: `record_count` counts every row ever inserted.
/// Mirrors the SQLite writer.
fn retire_prior_generation(tx: &Transaction<'_>, table_id: i64, snapshot_id: i64) -> Result<()> {
    tx.execute(
        "UPDATE ducklake_data_file SET end_snapshot = ?
         WHERE table_id = ? AND end_snapshot IS NULL AND begin_snapshot < ?",
        params![snapshot_id, table_id, snapshot_id],
    )?;
    for (table_name, system) in inlined_tables(tx, table_id)? {
        tx.execute(
            &format!(
                "UPDATE {} SET {end} = ? WHERE {end} IS NULL AND {begin} < ?",
                quote_ident(&table_name),
                end = system.end_snapshot,
                begin = system.begin_snapshot,
            ),
            params![snapshot_id, snapshot_id],
        )?;
    }
    Ok(())
}

/// The atomic commit point for a single-catalog write. Inserts the deferred
/// `ducklake_snapshot` row, finalizes the column generation surgically (stable
/// `column_id`s), classifies the commit as DDL vs pure data write to bump/carry
/// `schema_version`, and — for `Replace` — fences on the base snapshot and
/// retires the prior data generation. A faithful port of the SQLite writer's
/// `finalize_snapshot`; the only structural change is reading the current
/// columns into an owned `Vec` (dropping the prepared statement) before issuing
/// further writes on the same connection.
/// Persist the harvested per-column stats for a just-registered data file
/// (per-file zone maps). See the SQLite writer's equivalent for the rationale.
fn insert_file_column_stats(
    tx: &Transaction<'_>,
    table_id: i64,
    data_file_id: i64,
    column_stats: &[ColumnStat],
) -> Result<()> {
    crate::metadata_writer::reject_duplicate_column_stats(column_stats)?;
    for stat in column_stats {
        tx.execute(
            "INSERT INTO ducklake_file_column_stats
                 (data_file_id, table_id, column_id, column_size_bytes,
                  value_count, null_count, min_value, max_value, contains_nan, extra_stats)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
            params![
                data_file_id,
                table_id,
                stat.column_id,
                stat.column_size_bytes,
                stat.value_count,
                stat.null_count,
                stat.min_value.as_deref(),
                stat.max_value.as_deref(),
                stat.contains_nan,
            ],
        )?;
    }
    Ok(())
}

/// Persist a partitioned data file's partition metadata within the commit
/// transaction: set `ducklake_data_file.partition_id` and insert one
/// `ducklake_file_partition_value` row per partition key. A no-op for an
/// unpartitioned file.
fn insert_partition_metadata(
    tx: &Transaction<'_>,
    table_id: i64,
    data_file_id: i64,
    file: &DataFileInfo,
) -> Result<()> {
    if let Some(partition_id) = file.partition_id {
        tx.execute(
            "UPDATE ducklake_data_file SET partition_id = ? WHERE data_file_id = ?",
            params![partition_id, data_file_id],
        )?;
    }
    for (key_index, value) in &file.partition_values {
        tx.execute(
            "INSERT INTO ducklake_file_partition_value
                 (data_file_id, table_id, partition_key_index, partition_value)
             VALUES (?, ?, ?, ?)",
            params![data_file_id, table_id, i64::from(*key_index), value.as_deref()],
        )?;
    }
    Ok(())
}

/// The field ids and DuckLake types of the table's live columns and nested
/// fields, by which the table-wide statistics merge compares bounds.
fn live_stat_fields(tx: &Transaction<'_>, table_id: i64) -> Result<Vec<(i64, String)>> {
    let mut statement = tx.prepare(
        "SELECT column_id, column_type FROM ducklake_column
         WHERE table_id = ? AND end_snapshot IS NULL",
    )?;
    let fields = statement
        .query_map(params![table_id], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<duckdb::Result<Vec<_>>>()?;
    Ok(fields)
}

/// Merge the statistics of `files`, newly committed to the table, into its
/// `ducklake_table_column_stats` roll-up, as official DuckLake does when it
/// commits data files (see [`crate::stats_encode::merge_file_column_stats`]).
/// Called before the write advances `ducklake_table_stats`, which says whether
/// the table had statistics before it.
fn merge_file_table_column_stats(
    tx: &Transaction<'_>,
    table_id: i64,
    fields: &[(i64, String)],
    files: &[Vec<crate::stats_encode::FileColumnStat>],
) -> Result<()> {
    let initialized = table_stats_initialized(tx, table_id)?;
    let live: Vec<i64> = fields.iter().map(|(id, _)| *id).collect();
    let existing = load_table_column_stats(tx, table_id)?;
    let merged =
        crate::stats_encode::merge_file_column_stats(existing, initialized, files, &live, |id| {
            crate::metadata_writer::stat_kind_of(fields, id)
        });
    store_table_column_stats(tx, table_id, merged)
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
fn load_table_column_stats(
    tx: &Transaction<'_>,
    table_id: i64,
) -> Result<Vec<crate::stats_encode::GlobalColumnStat>> {
    let mut statement = tx.prepare(
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
    )?;
    Ok(statement
        .query_map(params![table_id], |row| {
            Ok(crate::stats_encode::GlobalColumnStat {
                column_id: row.get(0)?,
                contains_null: row.get(1)?,
                contains_nan: row.get(2)?,
                min_value: row.get(3)?,
                max_value: row.get(4)?,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?)
}

/// Replace the table's `ducklake_table_column_stats` rows with `globals`.
fn store_table_column_stats(
    tx: &Transaction<'_>,
    table_id: i64,
    globals: Vec<crate::stats_encode::GlobalColumnStat>,
) -> Result<()> {
    // A kept row keeps its `extra_stats` (a `GEOMETRY` column's bounds), which
    // this crate does not compute.
    let mut statement = tx.prepare(
        "SELECT column_id, extra_stats FROM ducklake_table_column_stats
         WHERE table_id = ? AND extra_stats IS NOT NULL",
    )?;
    let extra = statement
        .query_map(params![table_id], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?
        .collect::<std::result::Result<std::collections::HashMap<_, _>, _>>()?;
    tx.execute(
        "DELETE FROM ducklake_table_column_stats WHERE table_id = ?",
        params![table_id],
    )?;
    for g in globals {
        tx.execute(
            "INSERT INTO ducklake_table_column_stats
                 (table_id, column_id, contains_null, contains_nan, min_value, max_value, extra_stats)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![
                table_id,
                g.column_id,
                g.contains_null,
                g.contains_nan,
                g.min_value,
                g.max_value,
                extra.get(&g.column_id),
            ],
        )?;
    }
    Ok(())
}

/// Whether the table's statistics were initialized before this write, as
/// official keys an inlined write's statistics on an existing
/// `ducklake_table_stats` row. This crate seeds that row with zeros when it
/// creates a table, so a seeded row counts once it has recorded a row
/// (`next_row_id > 0`) or the table has column statistics.
fn table_stats_initialized(tx: &Transaction<'_>, table_id: i64) -> Result<bool> {
    Ok(tx.query_row(
        "SELECT EXISTS (SELECT 1 FROM ducklake_table_stats WHERE table_id = ? AND next_row_id > 0)
             OR EXISTS (SELECT 1 FROM ducklake_table_column_stats WHERE table_id = ?)",
        params![table_id, table_id],
        |row| row.get(0),
    )?)
}

/// Merge the statistics of `batches`, written inline to the table, into its
/// `ducklake_table_column_stats`, as official DuckLake does on an inlined write
/// (see [`crate::stats_encode::merge_inlined_column_stats`]). A NULL in one of
/// `snapshot_id_columns` is stored as `snapshot_id`.
#[expect(
    clippy::too_many_arguments,
    reason = "the statistics merge needs the table, its fields, the rows and the stats state"
)]
fn merge_inlined_table_column_stats(
    tx: &Transaction<'_>,
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
    let existing = load_table_column_stats(tx, table_id)?;
    let merged =
        crate::stats_encode::merge_inlined_column_stats(existing, inlined, initialized, |id| {
            crate::metadata_writer::stat_kind_of(&fields, id)
        });
    store_table_column_stats(tx, table_id, merged)
}

/// Official's table-level conflict check for a staged insert, `DELETE` or
/// `UPDATE` against every snapshot committed after its base snapshot and before
/// `snapshot_id` (see [`staged_conflict`]).
fn check_staged_conflicts(
    tx: &Transaction<'_>,
    write: &StagedTableWrite,
    snapshot_id: i64,
) -> Result<()> {
    let changes = changes_between(tx, write.base_snapshot_id, snapshot_id)?;
    staged_conflict(write, &changes).map_or(Ok(()), Err)
}

/// The `changes_made` of every snapshot after `base_snapshot` and before
/// `snapshot_id`, the commit's own.
fn changes_between(
    tx: &Transaction<'_>,
    base_snapshot: i64,
    snapshot_id: i64,
) -> Result<Vec<Option<String>>> {
    let mut statement = tx.prepare(
        "SELECT changes_made FROM ducklake_snapshot_changes
         WHERE snapshot_id > ? AND snapshot_id < ?",
    )?;
    Ok(statement
        .query_map(params![base_snapshot, snapshot_id], |row| {
            row.get::<_, Option<String>>(0)
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?)
}

/// The table the data files `deletes` delete from belong to: the table the
/// statement read, which a concurrent drop may have replaced under its name.
fn deleted_files_table(
    tx: &Transaction<'_>,
    table_id: i64,
    deletes: &[DeleteFileEntry],
) -> Result<i64> {
    let Some(entry) = deletes.first() else {
        return Ok(table_id);
    };
    let source: Option<i64> = tx
        .query_row(
            "SELECT table_id FROM ducklake_data_file WHERE data_file_id = ?",
            params![entry.data_file_id],
            |row| row.get(0),
        )
        .optional()?;
    Ok(source.unwrap_or(table_id))
}

/// Official's table-level conflict check for a commit at `snapshot_id` making
/// `ours` to `table_id` (see [`table_conflict`]).
fn check_table_conflicts(
    tx: &Transaction<'_>,
    table_id: i64,
    base_snapshot: i64,
    snapshot_id: i64,
    ours: TableChanges,
) -> Result<()> {
    let changes = changes_between(tx, base_snapshot, snapshot_id)?;
    table_conflict(table_id, ours, &changes).map_or(Ok(()), Err)
}

fn apply_delete_entry(
    tx: &Transaction<'_>,
    table_id: i64,
    base_snapshot: i64,
    snapshot_id: i64,
    entry: &DeleteFileEntry,
) -> Result<()> {
    let target_live: Option<i64> = tx
        .query_row(
            "SELECT 1 FROM ducklake_data_file
             WHERE data_file_id = ? AND end_snapshot IS NULL",
            params![entry.data_file_id],
            |row| row.get(0),
        )
        .optional()?;
    if target_live.is_none() {
        return Err(crate::DuckLakeError::Conflict(format!(
            "UPDATE/DELETE on data file {} could not commit: the file is no longer live as of \
             the catalog's current head (retired since snapshot {base_snapshot}); re-open the \
             catalog at the latest snapshot and retry",
            entry.data_file_id
        )));
    }
    let current_delete: Option<i64> = tx
        .query_row(
            "SELECT delete_file_id FROM ducklake_delete_file
             WHERE data_file_id = ? AND end_snapshot IS NULL",
            params![entry.data_file_id],
            |row| row.get(0),
        )
        .optional()?;
    if current_delete != entry.expected_prev_delete_file {
        return Err(crate::DuckLakeError::Conflict(format!(
            "UPDATE/DELETE on data file {} could not commit: its live delete file changed from \
             {:?} to {current_delete:?} since snapshot {base_snapshot}; re-open the catalog at \
             the latest snapshot and retry",
            entry.data_file_id, entry.expected_prev_delete_file
        )));
    }
    if let Some(delete_file_id) = entry.expected_prev_delete_file {
        tx.execute(
            "UPDATE ducklake_delete_file SET end_snapshot = ?
             WHERE delete_file_id = ? AND end_snapshot IS NULL",
            params![snapshot_id, delete_file_id],
        )?;
    }
    let delete_file_id = reserve_delete_file_ids(tx, 1)?[0];
    tx.execute(
        "INSERT INTO ducklake_delete_file
             (delete_file_id, data_file_id, table_id, path, path_is_relative,
              file_size_bytes, footer_size, delete_count, begin_snapshot)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            delete_file_id,
            entry.data_file_id,
            table_id,
            entry.delete.path.as_str(),
            entry.delete.path_is_relative,
            entry.delete.file_size_bytes,
            entry.delete.footer_size,
            entry.delete.delete_count,
            snapshot_id
        ],
    )?;
    Ok(())
}

/// The physical `ducklake_inlined_data_*` tables registered for the table, each
/// with its system column names. Mirrors the SQLite writer's `inlined_tables`.
fn inlined_tables(
    tx: &Transaction<'_>,
    table_id: i64,
) -> Result<Vec<(String, InlinedSystemColumns)>> {
    let mut statement =
        tx.prepare("SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?")?;
    let names = statement
        .query_map(params![table_id], |row| row.get::<_, String>(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    drop(statement);
    names
        .into_iter()
        .map(|name| {
            let system = inlined_table_layout(tx, &name)?.system;
            Ok((name, system))
        })
        .collect()
}

/// Count the table's still-visible inlined rows across its registered physical
/// tables. Mirrors the SQLite writer's `live_inlined_row_count`.
fn live_inlined_row_count(tx: &Transaction<'_>, table_id: i64) -> Result<i64> {
    let mut total = 0;
    for (table_name, system) in inlined_tables(tx, table_id)? {
        let sql = format!(
            "SELECT COUNT(*) FROM {} WHERE {} IS NULL",
            quote_ident(&table_name),
            system.end_snapshot,
        );
        total += tx.query_row(&sql, [], |row| row.get::<_, i64>(0))?;
    }
    Ok(total)
}

/// End the referenced inlined rows at `snapshot_id`, fencing each against
/// `base_snapshot`. Mirrors the SQLite writer's `apply_inlined_deletes`
/// (used by the combined `commit_deletes` path).
fn apply_inlined_deletes(
    tx: &Transaction<'_>,
    table_id: i64,
    snapshot_id: i64,
    base_snapshot: i64,
    rows: &[InlinedRowRef],
) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let registered = inlined_tables(tx, table_id)?;
    for (name, system) in &registered {
        let sql = format!(
            "SELECT EXISTS(SELECT 1 FROM {} WHERE ({begin} > ? AND {begin} < ?) OR ({end} > ? AND {end} < ?))",
            quote_ident(name),
            begin = system.begin_snapshot,
            end = system.end_snapshot,
        );
        let changed: bool = tx.query_row(
            &sql,
            params![base_snapshot, snapshot_id, base_snapshot, snapshot_id],
            |row| row.get(0),
        )?;
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
        tx.execute(&sql, params![snapshot_id, snapshot_id])?;
    }
    Ok(())
}

fn finalize_snapshot(
    tx: &Transaction<'_>,
    table_id: i64,
    columns: &[ColumnDef],
    column_ids: &[i64],
    mode: WriteMode,
    base_snapshot: i64,
) -> Result<i64> {
    let (snapshot_id, _) = insert_snapshot(tx)?;
    finalize_table_snapshot(
        tx,
        snapshot_id,
        table_id,
        columns,
        column_ids,
        mode,
        base_snapshot,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn finalize_table_snapshot(
    tx: &Transaction<'_>,
    snapshot_id: i64,
    table_id: i64,
    columns: &[ColumnDef],
    column_ids: &[i64],
    mode: WriteMode,
    base_snapshot: i64,
    inserts: bool,
) -> Result<i64> {
    use std::collections::{HashMap, HashSet};
    let proposed = catalog_column_defs(columns)?;
    if proposed.len() != column_ids.len() {
        return Err(crate::DuckLakeError::InvalidConfig(format!(
            "column_ids has {} entries for {} catalog column nodes",
            column_ids.len(),
            proposed.len()
        )));
    }

    // Allocate the snapshot FIRST (carrying schema_version forward); corrected to
    // a DDL bump below once the commit is classified.
    let mut schema_version: i64 = tx.query_row(
        "SELECT schema_version FROM ducklake_snapshot WHERE snapshot_id = ?",
        params![snapshot_id],
        |row| row.get(0),
    )?;

    // The table's live columns ordered by column_order. Collected into an owned
    // Vec so the prepared statement is dropped before we mutate the same
    // connection. An empty set means a brand-new table (its creating write is DDL).
    let current: Vec<(i64, String, String, i64, bool, Option<i64>)> = {
        let mut stmt = tx.prepare(
            "SELECT column_id, column_name, column_type, column_order, nulls_allowed, parent_column
             FROM ducklake_column
             WHERE table_id = ? AND end_snapshot IS NULL
             ORDER BY column_order",
        )?;
        let mapped = stmt.query_map(params![table_id], |row| {
            let column_id: i64 = row.get(0)?;
            let name: String = row.get(1)?;
            let ty: String = row.get(2)?;
            let order: i64 = row.get(3)?;
            let nullable: Option<bool> = row.get(4)?;
            let parent_column: Option<i64> = row.get(5)?;
            Ok((
                column_id,
                name,
                ty,
                order,
                nullable.unwrap_or(true),
                parent_column,
            ))
        })?;
        mapped.collect::<std::result::Result<Vec<_>, duckdb::Error>>()?
    };
    let existing_catalog_columns = current
        .iter()
        .map(|column| ExistingCatalogColumn {
            column_id: column.0,
            name: column.1.clone(),
            ducklake_type: column.2.clone(),
            parent_column: column.5,
        })
        .collect::<Vec<_>>();
    let existing_nullability = current.iter().map(|column| column.4).collect::<Vec<_>>();
    let committed_ids = assign_column_ids(&proposed, &existing_catalog_columns, column_ids)?;
    if committed_ids != column_ids {
        return Err(crate::DuckLakeError::Conflict(
            "table columns were created concurrently with different field ids; retry the write"
                .to_string(),
        ));
    }

    if current.is_empty() {
        tx.execute("UPDATE ducklake_schema SET begin_snapshot = ? WHERE schema_id = (SELECT schema_id FROM ducklake_table WHERE table_id = ?) AND begin_snapshot = (SELECT begin_snapshot FROM ducklake_table WHERE table_id = ?)", params![snapshot_id, table_id, table_id])?;
        tx.execute(
            "UPDATE ducklake_table SET begin_snapshot = ? WHERE table_id = ?",
            params![snapshot_id, table_id],
        )?;
    }
    let is_ddl = current.is_empty()
        || catalog_columns_differ(
            &existing_catalog_columns,
            &existing_nullability,
            &proposed,
            column_ids,
        );
    if is_ddl {
        reject_new_reserved_columns(tx, table_id, &existing_catalog_columns, &proposed)?;
    }
    if is_ddl {
        // A DDL commit bumps schema_version (the insert only carried it forward).
        schema_version = bump_schema_version(tx, snapshot_id)?;
    }

    // Reconcile the column generation SURGICALLY so each column keeps a stable
    // column_id (== parquet field_id): end only removed columns, insert only new
    // ones, and leave unchanged columns (and their ids) in place.
    let proposed_ids = column_ids.iter().copied().collect::<HashSet<_>>();
    let mut current_by_id: HashMap<i64, (i64, bool, String)> = HashMap::new();
    for (column_id, _name, ty, order, nullable, _parent_id) in &current {
        if !proposed_ids.contains(column_id) {
            tx.execute(
                "UPDATE ducklake_column SET end_snapshot = ?
                 WHERE table_id = ? AND column_id = ? AND end_snapshot IS NULL",
                params![snapshot_id, table_id, column_id],
            )?;
        }
        current_by_id.insert(*column_id, (*order, *nullable, ty.clone()));
    }

    for (order, (column, column_id)) in proposed.iter().zip(column_ids).enumerate() {
        let parent_id = column.parent_index.map(|index| column_ids[index]);
        match current_by_id.get(column_id) {
            // Existing column kept: its id stays stable. Sync metadata only if it
            // changed. A legacy list representation gets a snapshot-bounded row.
            Some((cur_order, cur_nullable, cur_type)) => {
                let migrate_type = catalog_column_type_requires_migration(cur_type, column);
                if migrate_type {
                    tx.execute(
                        "UPDATE ducklake_column SET end_snapshot = ?
                         WHERE table_id = ? AND column_id = ? AND end_snapshot IS NULL",
                        params![snapshot_id, table_id, column_id],
                    )?;
                    tx.execute(
                        "INSERT INTO ducklake_column
                             (column_id, table_id, column_name, column_type, column_order,
                              nulls_allowed, parent_column, begin_snapshot)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        params![
                            column_id,
                            table_id,
                            column.name.as_str(),
                            column.ducklake_type.as_str(),
                            order as i64,
                            column.is_nullable,
                            parent_id,
                            snapshot_id
                        ],
                    )?;
                } else if *cur_order != order as i64 || *cur_nullable != column.is_nullable {
                    tx.execute(
                        "UPDATE ducklake_column
                         SET column_order = ?, nulls_allowed = ?
                         WHERE table_id = ? AND column_id = ? AND end_snapshot IS NULL",
                        params![order as i64, column.is_nullable, table_id, column_id],
                    )?;
                }
            },
            // Newly added column: insert it with its reserved id.
            None => {
                tx.execute(
                    "INSERT INTO ducklake_column
                          (column_id, table_id, column_name, column_type, column_order,
                           nulls_allowed, parent_column, begin_snapshot, initial_default,
                           default_value, default_value_type, default_value_dialect)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    params![
                        column_id,
                        table_id,
                        column.name.as_str(),
                        column.ducklake_type.as_str(),
                        order as i64,
                        column.is_nullable,
                        parent_id,
                        snapshot_id,
                        column.initial_default.as_deref(),
                        column.default_value.as_deref(),
                        column.default_value_type.as_deref(),
                        column.default_value_dialect.as_deref()
                    ],
                )?;
            },
        }
    }

    if mode == WriteMode::Replace {
        // Abort if a concurrent writer published a newer generation since begin.
        detect_replace_conflict(tx, table_id, base_snapshot, ConflictScope::Table)?;
        // Retire the prior data generation. A table gets its stats row with
        // its first rows, as official's `CREATE TABLE` writes none.
        retire_prior_generation(tx, table_id, snapshot_id)?;
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
        )?;
    }

    // Record the schema-change ledger row for a DDL commit. A pure data write
    // carries schema_version forward and writes no row.
    if is_ddl {
        record_schema_version(tx, snapshot_id, schema_version, table_id)?;
        create_eager_inlined_table(tx, table_id, schema_version)?;
    }
    Ok(snapshot_id)
}

fn validate_staged_table(tx: &Transaction<'_>, write: &StagedTableWrite) -> Result<i64> {
    let schema_id: i64 = tx
        .query_row(
            "SELECT schema_id FROM ducklake_table
             WHERE table_id = ? AND end_snapshot IS NULL",
            params![write.table_id],
            |row| row.get(0),
        )
        .optional()?
        .ok_or_else(|| {
            crate::DuckLakeError::Conflict(format!(
                "multi-table write target {}.{} is no longer live",
                write.schema_name, write.table_name
            ))
        })?;
    let proposed = catalog_column_defs(&write.columns)?;
    let mut statement = tx.prepare(
        "SELECT column_id, column_type, nulls_allowed, parent_column
         FROM ducklake_column
         WHERE table_id = ? AND end_snapshot IS NULL
         ORDER BY column_order",
    )?;
    let current = statement
        .query_map(params![write.table_id], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<bool>>(2)?.unwrap_or(true),
                row.get::<_, Option<i64>>(3)?,
            ))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    drop(statement);
    if current.is_empty() && proposed.len() == write.column_ids.len() {
        return Ok(schema_id);
    }
    if current.len() != proposed.len() || proposed.len() != write.column_ids.len() {
        return Err(crate::DuckLakeError::Conflict(format!(
            "multi-table write target {}.{} changed schema after staging: catalog has {} nodes, \
             staged schema has {} nodes and {} field IDs",
            write.schema_name,
            write.table_name,
            current.len(),
            proposed.len(),
            write.column_ids.len(),
        )));
    }
    for (index, ((column_id, column_type, nullable, parent_id), proposed)) in
        current.iter().zip(&proposed).enumerate()
    {
        let proposed_parent = proposed.parent_index.map(|parent| write.column_ids[parent]);
        if *column_id != write.column_ids[index]
            || !catalog_column_type_equal(column_type, proposed)
            || *nullable != proposed.is_nullable
            || *parent_id != proposed_parent
        {
            return Err(crate::DuckLakeError::Conflict(format!(
                "multi-table write target {}.{} changed schema after staging at node {index}: \
                 catalog ({column_id}, {column_type}, nullable={nullable}, parent={parent_id:?}), \
                 staged ({}, {}, nullable={}, parent={proposed_parent:?})",
                write.schema_name,
                write.table_name,
                write.column_ids[index],
                proposed.ducklake_type,
                proposed.is_nullable,
            )));
        }
    }
    Ok(schema_id)
}

fn has_live_data(tx: &Transaction<'_>, table_id: i64) -> Result<(bool, bool)> {
    let has_files: bool = tx.query_row(
        "SELECT EXISTS(
             SELECT 1 FROM ducklake_data_file
             WHERE table_id = ? AND end_snapshot IS NULL
         )",
        params![table_id],
        |row| row.get(0),
    )?;
    for (table_name, system) in inlined_tables(tx, table_id)? {
        let has_rows: bool = tx.query_row(
            &format!(
                "SELECT EXISTS(SELECT 1 FROM {} WHERE {} IS NULL)",
                quote_ident(&table_name),
                system.end_snapshot,
            ),
            [],
            |row| row.get(0),
        )?;
        if has_rows {
            return Ok((has_files, true));
        }
    }
    Ok((has_files, false))
}

fn commit_staged_files(
    tx: &Transaction<'_>,
    snapshot_id: i64,
    write: &StagedTableWrite,
    files: &[DataFileInfo],
) -> Result<()> {
    if files.is_empty() {
        return Err(crate::DuckLakeError::InvalidConfig(
            "multi-table file stage requires at least one file".to_string(),
        ));
    }
    let live_partition_id: Option<i64> = tx.query_row(
        "SELECT (SELECT partition_id FROM ducklake_partition_info
         WHERE table_id = ? AND end_snapshot IS NULL LIMIT 1)",
        params![write.table_id],
        |row| row.get(0),
    )?;
    for file in files {
        crate::metadata_writer::enforce_partition_fence(write.table_id, live_partition_id, file)?;
    }
    seed_stats_if_missing(tx, write.table_id)?;
    let mut next_row_id: i64 = tx.query_row(
        "SELECT next_row_id FROM ducklake_table_stats WHERE table_id = ?",
        params![write.table_id],
        |row| row.get(0),
    )?;
    let mut total_advance = 0i64;
    let mut total_records = 0i64;
    let mut total_bytes = 0i64;
    for file in files {
        let row_ids = crate::metadata_writer::allocate_row_id_start(next_row_id, file);
        let data_file_id: i64 = tx.query_row(
            "INSERT INTO ducklake_data_file
                 (table_id, path, path_is_relative, file_size_bytes,
                  footer_size, record_count, row_id_start, begin_snapshot)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING data_file_id",
            params![
                write.table_id,
                file.path.as_str(),
                file.path_is_relative,
                file.file_size_bytes,
                file.footer_size,
                file.record_count,
                row_ids.stored,
                snapshot_id
            ],
            |row| row.get(0),
        )?;
        insert_file_column_stats(tx, write.table_id, data_file_id, &file.column_stats)?;
        insert_partition_metadata(tx, write.table_id, data_file_id, file)?;
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
    )?;
    tx.execute(
        "UPDATE ducklake_table_stats
         SET next_row_id = next_row_id + ?,
             record_count = record_count + ?,
             file_size_bytes = file_size_bytes + ?
         WHERE table_id = ?",
        params![total_advance, total_records, total_bytes, write.table_id],
    )?;
    Ok(())
}

/// The live column rows of `table_id`, each with its parent column id.
fn live_table_columns(
    tx: &Transaction<'_>,
    table_id: i64,
) -> Result<Vec<(crate::metadata_provider::DuckLakeTableColumn, Option<i64>)>> {
    let mut statement = tx.prepare(
        "SELECT column_id, column_name, column_type, nulls_allowed, parent_column
         FROM ducklake_column
         WHERE table_id = ? AND end_snapshot IS NULL
         ORDER BY column_order",
    )?;
    let rows = statement
        .query_map(params![table_id], |row| {
            let nullable: Option<bool> = row.get(3)?;
            Ok((
                crate::metadata_provider::DuckLakeTableColumn::new(
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    nullable.unwrap_or(true),
                ),
                row.get::<_, Option<i64>>(4)?,
            ))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// Create the inlined data table `ducklake_inlined_data_{table_id}_{schema_version}`
/// holding `fields` after the system columns, as official's DDL does, and
/// register it. Returns its name.
fn create_inlined_data_table(
    tx: &Transaction<'_>,
    table_id: i64,
    schema_version: i64,
    fields: &[(String, DataType, String)],
) -> Result<String> {
    let physical_name = format!("ducklake_inlined_data_{table_id}_{schema_version}");
    let system = catalog_inlined_system_columns(tx)?;
    let mut ddl = format!(
        "CREATE TABLE IF NOT EXISTS {} ({} BIGINT, {} BIGINT, {} BIGINT",
        quote_ident(&physical_name),
        system.row_id,
        system.begin_snapshot,
        system.end_snapshot,
    );
    for (name, data_type, ducklake_type) in fields {
        ddl.push_str(", ");
        ddl.push_str(&quote_ident(name));
        ddl.push(' ');
        ddl.push_str(&inlined_duckdb_type(data_type, ducklake_type)?);
    }
    ddl.push(')');
    tx.execute(&ddl, [])?;
    ensure_duckdb_physical_indexes(tx, table_id, &physical_name)?;
    tx.execute(
        "INSERT INTO ducklake_inlined_data_tables (table_id, table_name, schema_version)
         SELECT ?, ?, ? WHERE NOT EXISTS (
             SELECT 1 FROM ducklake_inlined_data_tables
             WHERE table_id = ? AND schema_version = ?
         )",
        params![table_id, physical_name.as_str(), schema_version, table_id, schema_version],
    )?;
    Ok(physical_name)
}

/// The `data_inlining_row_limit` in effect for the live table `table_id`;
/// `0` when it is not live.
fn table_inlining_limit(tx: &Transaction<'_>, table_id: i64) -> Result<usize> {
    let Some(schema_id): Option<i64> = tx
        .query_row(
            "SELECT schema_id FROM ducklake_table WHERE table_id = ? AND end_snapshot IS NULL",
            params![table_id],
            |row| row.get(0),
        )
        .optional()?
    else {
        return Ok(0);
    };
    let mut statement = tx.prepare(
        "SELECT scope, scope_id, value FROM ducklake_metadata
         WHERE key = 'data_inlining_row_limit'",
    )?;
    let settings = statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
        .collect::<std::result::Result<Vec<(Option<String>, Option<i64>, String)>, _>>()?;
    crate::metadata_writer::effective_inlining_limit(&settings, schema_id, table_id)
}

/// Refuse a commit that creates `table_id` or adds a column to it named like
/// an inlined data table's system column while its inlining limit is above
/// `0`, as official refuses `CREATE TABLE` and `ADD COLUMN`.
fn reject_new_reserved_columns(
    tx: &Transaction<'_>,
    table_id: i64,
    existing: &[ExistingCatalogColumn],
    proposed: &[crate::metadata_writer::CatalogColumnDef],
) -> Result<()> {
    let system = catalog_inlined_system_columns(tx)?;
    let Some(column) = crate::metadata_writer::new_reserved_column(existing, proposed, system)
    else {
        return Ok(());
    };
    if table_inlining_limit(tx, table_id)? == 0 {
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
fn create_eager_inlined_table(
    tx: &Transaction<'_>,
    table_id: i64,
    schema_version: i64,
) -> Result<()> {
    if table_inlining_limit(tx, table_id)? == 0 {
        return Ok(());
    }
    let columns = live_table_columns(tx, table_id)?;
    let system = catalog_inlined_system_columns(tx)?;
    if let Some(fields) =
        crate::metadata_writer::inlinable_columns(columns, system, None, duckdb_type_inlines)?
    {
        create_inlined_data_table(tx, table_id, schema_version, &fields)?;
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
fn inlined_insert_table(
    tx: &Transaction<'_>,
    table_id: i64,
    snapshot_id: i64,
    fields: &[(String, DataType, String)],
) -> Result<String> {
    let latest: Option<String> = tx
        .query_row(
            "SELECT table_name FROM ducklake_inlined_data_tables
             WHERE table_id = ? ORDER BY schema_version DESC LIMIT 1",
            params![table_id],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(name) = latest {
        let mut statement = tx.prepare("SELECT name FROM pragma_table_info(?) ORDER BY cid")?;
        let present = statement
            .query_map(params![name.as_str()], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let matches = present.len() == fields.len() + 3
            && present[3..]
                .iter()
                .zip(fields)
                .all(|(stored, (name, _, _))| stored == name);
        if matches {
            ensure_duckdb_physical_indexes(tx, table_id, &name)?;
            return Ok(name);
        }
        let changed_now: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM ducklake_column
                           WHERE table_id = ? AND (begin_snapshot = ? OR end_snapshot = ?))",
            params![table_id, snapshot_id, snapshot_id],
            |row| row.get(0),
        )?;
        if !changed_now {
            return Err(crate::metadata_writer::inlined_table_shape_error(
                &name,
                present.len(),
                fields.len() + 3,
            ));
        }
    }
    tx.execute(
        "UPDATE ducklake_snapshot SET schema_version = schema_version + 1 WHERE snapshot_id = ?",
        params![snapshot_id],
    )?;
    let schema_version: i64 = tx.query_row(
        "SELECT schema_version FROM ducklake_snapshot WHERE snapshot_id = ?",
        params![snapshot_id],
        |row| row.get(0),
    )?;
    create_inlined_data_table(tx, table_id, schema_version, fields)
}

fn commit_staged_inline(
    tx: &Transaction<'_>,
    snapshot_id: i64,
    write: &StagedTableWrite,
    batches: &[RecordBatch],
) -> Result<()> {
    enforce_inline_partition_fence(tx, write.table_id, write.base_snapshot_id)?;
    let record_count: usize = batches.iter().map(RecordBatch::num_rows).sum();
    if record_count == 0 {
        return Err(crate::DuckLakeError::InvalidConfig(
            "multi-table inline stage requires at least one row".to_string(),
        ));
    }
    let schema = batches[0].schema();
    let fields = write
        .columns
        .iter()
        .zip(schema.fields())
        .map(|(column, field)| {
            (
                column.name().to_string(),
                field.data_type().clone(),
                column.ducklake_type().to_string(),
            )
        })
        .collect::<Vec<_>>();
    let physical_name = inlined_insert_table(tx, write.table_id, snapshot_id, &fields)?;
    let system = inlined_table_layout(tx, &physical_name)?.system;
    let stats_initialized = table_stats_initialized(tx, write.table_id)?;
    seed_stats_if_missing(tx, write.table_id)?;
    let mut row_id: i64 = tx.query_row(
        "SELECT next_row_id FROM ducklake_table_stats WHERE table_id = ?",
        params![write.table_id],
        |row| row.get(0),
    )?;
    let column_list = write
        .columns
        .iter()
        .map(|column| quote_ident(column.name()))
        .collect::<Vec<_>>()
        .join(", ");
    let mut written = 0usize;
    for batch in batches {
        for batch_row in 0..batch.num_rows() {
            let encoded = batch
                .columns()
                .iter()
                .zip(&write.columns)
                .zip(schema.fields())
                .map(|((array, column), field)| {
                    if write
                        .snapshot_id_columns
                        .iter()
                        .any(|name| name == column.name())
                        && array.is_null(batch_row)
                    {
                        Ok(EncodedDuckdbValue {
                            sql: inlined_duckdb_value_sql(
                                field.data_type(),
                                column.ducklake_type(),
                            )?,
                            binds: vec![Value::BigInt(snapshot_id)],
                        })
                    } else {
                        encode_duckdb_value(array.as_ref(), batch_row, column.ducklake_type(), 0)
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            let value_list = encoded
                .iter()
                .map(|value| value.sql.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            let insert_sql = format!(
                "INSERT INTO {} ({}, {}, {}, {}) \
                 VALUES (?, ?, ?, {})",
                quote_ident(&physical_name),
                system.row_id,
                system.begin_snapshot,
                system.end_snapshot,
                column_list,
                value_list
            );
            let stored_row_id = match &write.inlined_row_ids {
                Some(row_ids) => row_ids[written],
                None => row_id,
            };
            let mut values = Vec::new();
            values.push(Value::BigInt(stored_row_id));
            values.push(Value::BigInt(snapshot_id));
            values.push(Value::Null);
            for value in encoded {
                values.extend(value.binds);
            }
            tx.execute(&insert_sql, params_from_iter(values))?;
            row_id += 1;
            written += 1;
        }
    }
    merge_inlined_table_column_stats(
        tx,
        write.table_id,
        &write.columns,
        &write.column_ids,
        batches,
        &write.snapshot_id_columns,
        snapshot_id,
        stats_initialized,
    )?;
    let record_count = i64::try_from(record_count).map_err(|_| {
        crate::DuckLakeError::InvalidConfig("multi-table inline row count exceeds i64".to_string())
    })?;
    // Preserved row ids were drawn when the rows were first written.
    let advance = if write.inlined_row_ids.is_some() {
        0
    } else {
        record_count
    };
    tx.execute(
        "UPDATE ducklake_table_stats
         SET next_row_id = next_row_id + ?, record_count = record_count + ?
         WHERE table_id = ?",
        params![advance, record_count, write.table_id],
    )?;
    Ok(())
}

fn apply_staged_inlined_deletes(
    tx: &Transaction<'_>,
    snapshot_id: i64,
    write: &StagedTableWrite,
) -> Result<()> {
    apply_inlined_deletes(
        tx,
        write.table_id,
        snapshot_id,
        write.base_snapshot_id,
        &write.inlined_deletes,
    )
}

/// Record `deletes` in `ducklake_inlined_delete_<table_id>`, creating it as
/// official DuckLake does. Aborts when a target data file is no longer live;
/// the caller has checked the table-level conflicts ([`table_conflict`]).
fn apply_inlined_file_deletes(
    tx: &Transaction<'_>,
    table_id: i64,
    snapshot_id: i64,
    base_snapshot: i64,
    deletes: &[InlinedFileDelete],
) -> Result<()> {
    if deletes.is_empty() {
        return Ok(());
    }
    for delete in deletes {
        let live: Option<i64> = tx
            .query_row(
                "SELECT 1 FROM ducklake_data_file
                 WHERE data_file_id = ? AND table_id = ? AND end_snapshot IS NULL",
                params![delete.data_file_id, table_id],
                |row| row.get(0),
            )
            .optional()?;
        if live.is_none() {
            return Err(crate::DuckLakeError::Conflict(format!(
                "data file {} is no longer live since snapshot {base_snapshot}; re-open the \
                 catalog and retry the statement",
                delete.data_file_id
            )));
        }
        let delete_file: Option<i64> = tx
            .query_row(
                "SELECT delete_file_id FROM ducklake_delete_file
                 WHERE data_file_id = ? AND end_snapshot IS NULL",
                params![delete.data_file_id],
                |row| row.get(0),
            )
            .optional()?;
        if delete_file != delete.expected_delete_file {
            return Err(crate::metadata_writer::changed_delete_file_conflict(
                delete.data_file_id,
                base_snapshot,
            ));
        }
    }
    let table = crate::metadata_provider::inlined_delete_table_name(table_id)?;
    tx.execute(
        &format!(
            "CREATE TABLE IF NOT EXISTS {} (file_id BIGINT, row_id BIGINT, begin_snapshot BIGINT)",
            quote_ident(&table)
        ),
        [],
    )?;
    let insert = format!(
        "INSERT INTO {} (file_id, row_id, begin_snapshot) VALUES (?, ?, ?)",
        quote_ident(&table)
    );
    let mut statement = tx.prepare(&insert)?;
    for delete in deletes {
        for position in &delete.positions {
            statement.execute(params![delete.data_file_id, position, snapshot_id])?;
        }
    }
    Ok(())
}

/// End each of `files` at `snapshot_id`, leaving its live delete file as it
/// is, as official's `DropDataFiles` does. Aborts when a file is no longer
/// live or its live delete file changed.
fn drop_data_files(
    tx: &Transaction<'_>,
    table_id: i64,
    snapshot_id: i64,
    base_snapshot: i64,
    files: &[DroppedDataFile],
) -> Result<()> {
    for file in files {
        let current: Option<i64> = tx
            .query_row(
                "SELECT delete_file_id FROM ducklake_delete_file
                 WHERE data_file_id = ? AND end_snapshot IS NULL",
                params![file.data_file_id],
                |row| row.get(0),
            )
            .optional()?;
        let ended = tx.execute(
            "UPDATE ducklake_data_file SET end_snapshot = ?
             WHERE data_file_id = ? AND table_id = ? AND end_snapshot IS NULL",
            params![snapshot_id, file.data_file_id, table_id],
        )?;
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

impl DuckdbMetadataWriter {
    pub(crate) fn list_scheduled_for_deletion(
        &self,
        criteria: &CleanupCriteria,
    ) -> Result<Vec<ScheduledFile>> {
        let conn = self.connection();
        let (sql, cutoff) = match criteria {
            CleanupCriteria::All => (
                "SELECT data_file_id, path, path_is_relative \
                 FROM ducklake_files_scheduled_for_deletion",
                None,
            ),
            CleanupCriteria::OlderThan(cutoff) => (
                "SELECT data_file_id, path, path_is_relative \
                 FROM ducklake_files_scheduled_for_deletion \
                 WHERE schedule_start < CAST(? AS TIMESTAMP)",
                Some(format_sql_timestamp(cutoff)),
            ),
        };
        let mut statement = conn.prepare(sql)?;
        let decode = |row: &duckdb::Row<'_>| {
            Ok(ScheduledFile {
                data_file_id: row.get(0)?,
                path: row.get(1)?,
                path_is_relative: row.get(2)?,
                // Single-catalog layout — see the SQLite writer.
                scheduled_by_owner: None,
            })
        };
        let files = if let Some(cutoff) = cutoff {
            statement
                .query_map([cutoff], decode)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            statement
                .query_map([], decode)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        Ok(files)
    }

    pub(crate) fn remove_scheduled(&self, ids: &[i64]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let conn = self.connection();
        conn.execute_batch(&format!(
            "DELETE FROM ducklake_files_scheduled_for_deletion \
             WHERE data_file_id IN ({})",
            id_list(ids),
        ))?;
        Ok(())
    }

    pub(crate) fn list_referenced_paths(&self) -> Result<Vec<(String, bool)>> {
        let conn = self.connection();
        let sql = format!(
            "SELECT {RESOLVED_PATH} AS path, {REL_FLAG} AS path_is_relative \
             FROM ducklake_data_file AS df \
             JOIN ducklake_table AS t ON t.table_id = df.table_id \
             JOIN ducklake_schema AS s ON s.schema_id = t.schema_id \
             UNION ALL \
             SELECT {RESOLVED_PATH} AS path, {REL_FLAG} AS path_is_relative \
             FROM ducklake_delete_file AS df \
             JOIN ducklake_table AS t ON t.table_id = df.table_id \
             JOIN ducklake_schema AS s ON s.schema_id = t.schema_id \
             UNION ALL \
             SELECT path, path_is_relative \
             FROM ducklake_files_scheduled_for_deletion",
        );
        let mut statement = conn.prepare(&sql)?;
        let paths = statement
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(paths)
    }
}

impl MetadataWriter for DuckdbMetadataWriter {
    fn maintenance(&self) -> Option<&dyn crate::maintenance::CatalogMaintenance> {
        Some(self)
    }

    fn supports_update(&self) -> bool {
        true
    }

    fn supports_inlined_row_mutations(&self) -> bool {
        true
    }

    fn drop_empty_superseded_inlined_tables(&self) -> Result<()> {
        let mut conn = self.connection();
        // One transaction: a registry row and its physical table go together.
        let tx = conn.transaction()?;
        // A catalog this crate's writer created before data inlining has no registry,
        // and so nothing to drop.
        if !self.metadata_provider().has_inlined_data_registry(&tx)? {
            return Ok(());
        }
        let targets = {
            let mut statement = tx.prepare(SUPERSEDED_INLINED_TABLES_SQL)?;
            statement
                .query_map([], |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        for (table_id, schema_version, table_name) in targets {
            let quoted = quote_ident(&table_name);
            let rows: i64 = tx.query_row(&format!("SELECT COUNT(*) FROM {quoted}"), [], |row| {
                row.get(0)
            })?;
            if rows != 0 {
                continue;
            }
            tx.execute(
                "DELETE FROM ducklake_inlined_data_tables
                 WHERE table_id = ? AND schema_version = ?",
                params![table_id, schema_version],
            )?;
            tx.execute_batch(&format!("DROP TABLE IF EXISTS {quoted}"))?;
        }
        tx.commit()?;
        Ok(())
    }

    fn catalog_encrypted(&self) -> Result<bool> {
        let connection = self.connection();
        let mut statement = connection.prepare(
            "SELECT value FROM ducklake_metadata WHERE key = 'encrypted' AND scope IS NULL",
        )?;
        let values = statement
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<duckdb::Result<Vec<_>>>()?;
        crate::metadata_writer::encrypted_option(&values)
    }

    fn head_snapshot(&self) -> Result<Option<i64>> {
        Ok(self.connection().query_row(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot",
            [],
            |row| row.get(0),
        )?)
    }

    fn create_snapshot(&self) -> Result<i64> {
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        // A bare snapshot carries no schema change of its own → carry
        // schema_version forward (no DDL bump, no ledger row).
        let (snapshot_id, _schema_version) = insert_snapshot(&tx)?;
        tx.commit()?;
        Ok(snapshot_id)
    }

    fn set_global_setting(&self, key: &str, value: &str) -> Result<()> {
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        if enables_inlining(key, value) {
            let columns = tx
                .prepare(SQL_GLOBAL_INLINING_COLUMNS)?
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<std::result::Result<Vec<(String, String)>, _>>()?;
            let system = crate::metadata_provider_duckdb::catalog_inlined_system_columns(&tx)?;
            reject_reserved_inlining_columns(&columns, system)?;
        }
        tx.execute(
            "DELETE FROM ducklake_metadata WHERE key = ? AND scope IS NULL",
            params![key],
        )?;
        tx.execute(
            "INSERT INTO ducklake_metadata (key, value, scope, scope_id) VALUES (?, ?, NULL, NULL)",
            params![key, value],
        )?;
        tx.commit()?;
        Ok(())
    }

    fn set_table_setting(&self, table_id: i64, key: &str, value: &str) -> Result<()> {
        let key = validate_table_setting(key)?;
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        let exists: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM ducklake_table WHERE table_id = ? AND end_snapshot IS NULL)",
            params![table_id],
            |row| row.get(0),
        )?;
        if !exists {
            return Err(crate::DuckLakeError::TableNotFound(table_id.to_string()));
        }
        if enables_inlining(&key, value) {
            let columns = tx
                .prepare(SQL_TABLE_INLINING_COLUMNS)?
                .query_map(params![table_id], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<std::result::Result<Vec<(String, String)>, _>>()?;
            let system = crate::metadata_provider_duckdb::catalog_inlined_system_columns(&tx)?;
            reject_reserved_inlining_columns(&columns, system)?;
        }
        tx.execute(
            "DELETE FROM ducklake_metadata WHERE key = ? AND scope = 'table' AND scope_id = ?",
            params![key, table_id],
        )?;
        tx.execute(
            "INSERT INTO ducklake_metadata (key, value, scope, scope_id) VALUES (?, ?, 'table', ?)",
            params![key, value, table_id],
        )?;
        tx.commit()?;
        Ok(())
    }

    fn set_inlined_index_columns(&self, table_id: i64, columns: &[String]) -> Result<()> {
        let value = encode_inlined_index_columns(columns)?;
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        let mut statement = tx.prepare(
            "SELECT column_name FROM ducklake_column
             WHERE table_id = ? AND end_snapshot IS NULL AND parent_column IS NULL",
        )?;
        let available = statement
            .query_map(params![table_id], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<std::collections::HashSet<_>, _>>()?;
        drop(statement);
        if available.is_empty() {
            return Err(crate::DuckLakeError::TableNotFound(table_id.to_string()));
        }
        validate_inlined_index_columns(columns, &available)?;
        let existing: Option<String> = tx
            .query_row(
                "SELECT value FROM ducklake_metadata
                 WHERE key = ? AND scope = 'table' AND scope_id = ?",
                params![INLINED_INDEX_COLUMNS_SETTING, table_id],
                |row| row.get(0),
            )
            .optional()?;
        if existing.as_deref() == Some(value.as_str()) {
            tx.commit()?;
            return Ok(());
        }
        tx.execute(
            "DELETE FROM ducklake_metadata
             WHERE key = ? AND scope = 'table' AND scope_id = ?",
            params![INLINED_INDEX_COLUMNS_SETTING, table_id],
        )?;
        tx.execute(
            "INSERT INTO ducklake_metadata (key, value, scope, scope_id)
             VALUES (?, ?, 'table', ?)",
            params![INLINED_INDEX_COLUMNS_SETTING, value, table_id],
        )?;
        tx.commit()?;
        Ok(())
    }

    fn ensure_inlined_indexes(&self, table_id: i64) -> Result<()> {
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        let mut statement =
            tx.prepare("SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?")?;
        let physical_tables = statement
            .query_map(params![table_id], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        drop(statement);
        duckdb_inlined_index_columns(&tx, table_id)?;
        for physical_table in physical_tables {
            ensure_duckdb_physical_indexes(&tx, table_id, &physical_table)?;
        }
        tx.commit()?;
        Ok(())
    }

    fn with_commit_lock(
        &self,
        _identity: &str,
        operation: Box<dyn FnOnce() -> Result<()> + '_>,
    ) -> Result<()> {
        if self.catalog_path == ":memory:" {
            let _guard = self.commit_lock.lock().map_err(|_| {
                crate::DuckLakeError::Internal("DuckDB commit lock is poisoned".to_string())
            })?;
            return operation();
        }

        let lock_path = format!("{}.commit.lock", self.catalog_path);
        let lock_file = OpenOptions::new()
            .create(true)
            .read(true)
            .truncate(false)
            .write(true)
            .open(lock_path)?;
        lock_file.lock()?;
        let result = operation();
        let unlock = lock_file.unlock();
        match (result, unlock) {
            (Ok(()), Ok(())) => Ok(()),
            (Ok(()), Err(e)) => Err(e.into()),
            (Err(e), _) => Err(e),
        }
    }

    fn get_or_create_schema(
        &self,
        name: &str,
        path: Option<&str>,
        snapshot_id: i64,
    ) -> Result<(i64, bool)> {
        validate_name(name, "Schema")?;
        let mut conn = self.connection();
        let tx = conn.transaction()?;

        let existing: Option<i64> = tx
            .query_row(
                "SELECT schema_id FROM ducklake_schema
                 WHERE schema_name = ? AND end_snapshot IS NULL",
                params![name],
                |row| row.get(0),
            )
            .optional()?;

        if let Some(schema_id) = existing {
            tx.commit()?;
            return Ok((schema_id, false));
        }

        let schema_path = directory_path(path.unwrap_or(name));
        let schema_id: i64 = tx.query_row(
            "INSERT INTO ducklake_schema (schema_name, path, path_is_relative, begin_snapshot)
             VALUES (?, ?, true, ?) RETURNING schema_id",
            params![name, schema_path, snapshot_id],
            |row| row.get(0),
        )?;
        record_snapshot_changes(
            &tx,
            snapshot_id,
            &format!("created_schema:{}", quote_snapshot_name(name)),
            &SnapshotCommitMetadata::default(),
        )?;
        tx.commit()?;
        Ok((schema_id, true))
    }

    fn get_or_create_table(
        &self,
        schema_id: i64,
        name: &str,
        path: Option<&str>,
        snapshot_id: i64,
    ) -> Result<(i64, bool)> {
        validate_name(name, "Table")?;
        let mut conn = self.connection();
        let tx = conn.transaction()?;

        let existing: Option<i64> = tx
            .query_row(
                "SELECT table_id FROM ducklake_table
                 WHERE schema_id = ? AND table_name = ? AND end_snapshot IS NULL",
                params![schema_id, name],
                |row| row.get(0),
            )
            .optional()?;

        if let Some(table_id) = existing {
            tx.commit()?;
            return Ok((table_id, false));
        }

        let schema_name: String = tx.query_row(
            "SELECT schema_name FROM ducklake_schema WHERE schema_id = ?",
            params![schema_id],
            |row| row.get(0),
        )?;

        let table_path = directory_path(path.unwrap_or(name));
        let table_id: i64 = tx.query_row(
            "INSERT INTO ducklake_table (schema_id, table_name, path, path_is_relative, begin_snapshot)
             VALUES (?, ?, ?, true, ?) RETURNING table_id",
            params![schema_id, name, table_path, snapshot_id],
            |row| row.get(0),
        )?;
        record_snapshot_changes(
            &tx,
            snapshot_id,
            &format!("created_table:{}", quote_snapshot_table(&schema_name, name)),
            &SnapshotCommitMetadata::default(),
        )?;
        tx.commit()?;
        Ok((table_id, true))
    }

    fn promote_column_type(
        &self,
        table_id: i64,
        column_name: &str,
        new_ducklake_type: &str,
    ) -> Result<i64> {
        crate::types::ducklake_to_arrow_type(new_ducklake_type)?;
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        let (snapshot_id, _carried) = insert_snapshot(&tx)?;
        // The new version keeps everything but the type: defaults included,
        // so rows written before the column was added still read its default.
        type LiveColumn = (i64, String, i64, Option<bool>, [Option<String>; 4]);
        let row: Option<LiveColumn> = tx
            .query_row(
                "SELECT column_id, column_type, column_order, nulls_allowed, initial_default,
                        default_value, default_value_type, default_value_dialect
                 FROM ducklake_column
                 WHERE table_id = ? AND column_name = ? AND end_snapshot IS NULL
                   AND parent_column IS NULL",
                params![table_id, column_name],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        [row.get(4)?, row.get(5)?, row.get(6)?, row.get(7)?],
                    ))
                },
            )
            .optional()?;
        let (column_id, current_type, column_order, nulls_allowed, defaults) =
            row.ok_or_else(|| {
                crate::DuckLakeError::InvalidConfig(format!(
                    "promote_column_type: no live column '{column_name}' in table {table_id}"
                ))
            })?;
        if crate::types::types_equal_canonical(&current_type, new_ducklake_type) {
            return Err(crate::DuckLakeError::InvalidConfig(format!(
                "promote_column_type: column '{column_name}' is already type '{current_type}' (no change)"
            )));
        }
        if !crate::types::is_promotable(&current_type, new_ducklake_type) {
            return Err(crate::DuckLakeError::UnsupportedTypeChange {
                operation: TypeChangeOperation::PromoteColumnType,
                column: column_name.to_string(),
                from: current_type,
                to: new_ducklake_type.to_string(),
            });
        }

        let schema_version = bump_schema_version(&tx, snapshot_id)?;
        tx.execute(
            "UPDATE ducklake_column SET end_snapshot = ?
             WHERE table_id = ? AND column_id = ? AND end_snapshot IS NULL",
            params![snapshot_id, table_id, column_id],
        )?;
        let [initial_default, default_value, default_value_type, default_value_dialect] = defaults;
        tx.execute(
            "INSERT INTO ducklake_column
                 (column_id, begin_snapshot, end_snapshot, table_id, column_order,
                  column_name, column_type, nulls_allowed, initial_default, default_value,
                  default_value_type, default_value_dialect)
             VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                column_id,
                snapshot_id,
                table_id,
                column_order,
                column_name,
                new_ducklake_type,
                nulls_allowed.unwrap_or(true),
                initial_default,
                default_value,
                default_value_type,
                default_value_dialect
            ],
        )?;
        record_schema_version(&tx, snapshot_id, schema_version, table_id)?;
        create_eager_inlined_table(&tx, table_id, schema_version)?;
        record_snapshot_changes(
            &tx,
            snapshot_id,
            &format!("altered_table:{table_id}"),
            &SnapshotCommitMetadata::default(),
        )?;
        tx.commit()?;
        Ok(snapshot_id)
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
        let mut conn = self.connection();
        let tx = conn.transaction()?;

        tx.execute(
            "UPDATE ducklake_column SET end_snapshot = ?
             WHERE table_id = ? AND end_snapshot IS NULL",
            params![snapshot_id, table_id],
        )?;

        // Reserve a contiguous column_id block from the monotonic counter and
        // insert with explicit ids, keeping the allocator authoritative.
        let catalog_columns = catalog_column_defs(columns)?;
        let n = catalog_columns.len() as i64;
        let last_column_id = reserve_ids(&tx, "next_column_id", n)?;
        let first_column_id = last_column_id - n + 1;
        let column_ids = (first_column_id..=last_column_id).collect::<Vec<_>>();
        for (order, (column, column_id)) in
            catalog_columns.iter().zip(column_ids.iter()).enumerate()
        {
            let parent_id = column.parent_index.map(|index| column_ids[index]);
            tx.execute(
                "INSERT INTO ducklake_column
                      (column_id, table_id, column_name, column_type, column_order,
                       nulls_allowed, parent_column, begin_snapshot, initial_default,
                       default_value, default_value_type, default_value_dialect)
                  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params![
                    column_id,
                    table_id,
                    column.name.as_str(),
                    column.ducklake_type.as_str(),
                    order as i64,
                    column.is_nullable,
                    parent_id,
                    snapshot_id,
                    column.initial_default.as_deref(),
                    column.default_value.as_deref(),
                    column.default_value_type.as_deref(),
                    column.default_value_dialect.as_deref()
                ],
            )?;
        }

        let table_begin_snapshot: i64 = tx.query_row(
            "SELECT begin_snapshot FROM ducklake_table WHERE table_id = ?",
            params![table_id],
            |row| row.get(0),
        )?;
        if table_begin_snapshot != snapshot_id {
            record_snapshot_changes(
                &tx,
                snapshot_id,
                &format!("altered_table:{table_id}"),
                &SnapshotCommitMetadata::default(),
            )?;
        }
        tx.commit()?;
        top_level_column_ids(&catalog_columns, &column_ids)
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
        if expected_base_snapshot_id.is_some() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "conditional writes are not supported by the DuckDB metadata writer".to_string(),
            ));
        }
        let mut conn = self.connection();
        let tx = conn.transaction()?;

        // Single atomic commit: insert the deferred snapshot row + finalize the
        // column generation + retire the prior generation (Replace), then
        // register this file and advance the monotonic row-lineage counter.
        let snapshot_id =
            finalize_snapshot(&tx, table_id, columns, column_ids, mode, base_snapshot)?;

        // Partition-spec fence: this file must be consistent with the table's live
        // partition generation at commit time (both directions — see
        // enforce_partition_fence). The scalar subquery yields NULL (→ None) when the
        // table is unpartitioned; the tx rolls back on a Conflict.
        let live_partition_id: Option<i64> = tx.query_row(
            "SELECT (SELECT partition_id FROM ducklake_partition_info
                     WHERE table_id = ? AND end_snapshot IS NULL LIMIT 1)",
            params![table_id],
            |row| row.get::<_, Option<i64>>(0),
        )?;
        crate::metadata_writer::enforce_partition_fence(table_id, live_partition_id, file)?;

        // Seed the stats row for the Append path (Replace already seeded it in
        // finalize_snapshot); a no-op if it exists.
        seed_stats_if_missing(&tx, table_id)?;

        let next_row_id: i64 = tx.query_row(
            "SELECT next_row_id FROM ducklake_table_stats WHERE table_id = ?",
            params![table_id],
            |row| row.get(0),
        )?;
        let row_ids = crate::metadata_writer::allocate_row_id_start(next_row_id, file);

        // RETURNING gives us the sequence-allocated data_file_id to tie the
        // per-column stats rows to, in the same transaction.
        let data_file_id: i64 = tx.query_row(
            "INSERT INTO ducklake_data_file
                 (table_id, path, path_is_relative, file_size_bytes,
                  footer_size, record_count, row_id_start, begin_snapshot)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING data_file_id",
            params![
                table_id,
                file.path.as_str(),
                file.path_is_relative,
                file.file_size_bytes,
                file.footer_size,
                file.record_count,
                row_ids.stored,
                snapshot_id
            ],
            |row| row.get(0),
        )?;

        insert_file_column_stats(&tx, table_id, data_file_id, &file.column_stats)?;
        insert_partition_metadata(&tx, table_id, data_file_id, file)?;
        merge_file_table_column_stats(
            &tx,
            table_id,
            &crate::metadata_writer::catalog_fields(columns, column_ids)?,
            &crate::metadata_writer::file_column_stats([file]),
        )?;

        // Advance the counter and accumulate stats. next_row_id monotonically
        // increases over the table's lifetime — rowids are never reused.
        tx.execute(
            "UPDATE ducklake_table_stats
             SET next_row_id     = next_row_id + ?,
                 record_count    = record_count + ?,
                 file_size_bytes = file_size_bytes + ?
             WHERE table_id = ?",
            params![row_ids.advance, file.record_count, file.file_size_bytes, table_id],
        )?;

        record_table_write_changes(
            &tx,
            snapshot_id,
            table_id,
            schema_name,
            table_name,
            mode,
            false,
            false,
            commit_metadata,
        )?;
        let schema_id: i64 = tx.query_row(
            "SELECT schema_id FROM ducklake_table WHERE table_id = ?",
            params![table_id],
            |row| row.get(0),
        )?;

        tx.commit()?;
        Ok(CommitIds {
            snapshot_id,
            schema_id,
            table_id,
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
        if expected_base_snapshot_id.is_some() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "conditional multi-file writes are not supported by the DuckDB metadata writer"
                    .to_string(),
            ));
        }
        if files.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "register_data_files: files must be non-empty".to_string(),
            ));
        }
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        // One atomic snapshot for all N partition files.
        let snapshot_id =
            finalize_snapshot(&tx, table_id, columns, column_ids, mode, base_snapshot)?;
        // Partition-spec fence (both directions, every file): each file must be
        // consistent with the table's live partition generation at commit time. The
        // scalar subquery yields NULL (→ None) when unpartitioned; the tx rolls back
        // on a Conflict.
        let live_partition_id: Option<i64> = tx.query_row(
            "SELECT (SELECT partition_id FROM ducklake_partition_info
                     WHERE table_id = ? AND end_snapshot IS NULL LIMIT 1)",
            params![table_id],
            |row| row.get::<_, Option<i64>>(0),
        )?;
        for file in files {
            crate::metadata_writer::enforce_partition_fence(table_id, live_partition_id, file)?;
        }
        seed_stats_if_missing(&tx, table_id)?;
        let mut next_row_id: i64 = tx.query_row(
            "SELECT next_row_id FROM ducklake_table_stats WHERE table_id = ?",
            params![table_id],
            |row| row.get(0),
        )?;
        let mut total_advance: i64 = 0;
        let mut total_records: i64 = 0;
        let mut total_bytes: i64 = 0;
        for file in files {
            let row_ids = crate::metadata_writer::allocate_row_id_start(next_row_id, file);
            let data_file_id: i64 = tx.query_row(
                "INSERT INTO ducklake_data_file
                     (table_id, path, path_is_relative, file_size_bytes,
                      footer_size, record_count, row_id_start, begin_snapshot)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING data_file_id",
                params![
                    table_id,
                    file.path.as_str(),
                    file.path_is_relative,
                    file.file_size_bytes,
                    file.footer_size,
                    file.record_count,
                    row_ids.stored,
                    snapshot_id
                ],
                |row| row.get(0),
            )?;
            insert_file_column_stats(&tx, table_id, data_file_id, &file.column_stats)?;
            insert_partition_metadata(&tx, table_id, data_file_id, file)?;
            next_row_id += row_ids.advance;
            total_advance += row_ids.advance;
            total_records += file.record_count;
            total_bytes += file.file_size_bytes;
        }
        merge_file_table_column_stats(
            &tx,
            table_id,
            &crate::metadata_writer::catalog_fields(columns, column_ids)?,
            &crate::metadata_writer::file_column_stats(files),
        )?;
        tx.execute(
            "UPDATE ducklake_table_stats
             SET next_row_id     = next_row_id + ?,
                 record_count    = record_count + ?,
                 file_size_bytes = file_size_bytes + ?
             WHERE table_id = ?",
            params![total_advance, total_records, total_bytes, table_id],
        )?;
        record_table_write_changes(
            &tx,
            snapshot_id,
            table_id,
            schema_name,
            table_name,
            mode,
            false,
            false,
            commit_metadata,
        )?;
        let schema_id: i64 = tx.query_row(
            "SELECT schema_id FROM ducklake_table WHERE table_id = ?",
            params![table_id],
            |row| row.get(0),
        )?;
        tx.commit()?;
        Ok(CommitIds {
            snapshot_id,
            schema_id,
            table_id,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn reserved_inlining_column(&self, columns: &[String]) -> Result<Option<String>> {
        let system = catalog_inlined_system_columns(&self.connection())?;
        Ok(columns
            .iter()
            .find(|column| system.conflicts_with(column))
            .cloned())
    }

    fn supports_data_inlining(&self, schema: &arrow::datatypes::Schema) -> bool {
        let Ok(system) = catalog_inlined_system_columns(&self.connection()) else {
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
            .all(|field| duckdb_type_inlines(field.data_type()))
    }

    fn supports_data_inlining_values(&self, batches: &[RecordBatch]) -> bool {
        batches
            .iter()
            .flat_map(RecordBatch::columns)
            .all(|array| crate::nested_inline::values_support_inlining(array.as_ref()))
    }

    fn table_types_support_inlining(&self, schema_name: &str, table_name: &str) -> Result<bool> {
        let refused: bool = self.connection().query_row(
            &format!(
                "SELECT EXISTS(
                     SELECT 1 FROM ducklake_column c
                     JOIN ducklake_table t ON t.table_id = c.table_id
                     JOIN ducklake_schema s ON s.schema_id = t.schema_id
                     WHERE s.schema_name = ? AND s.end_snapshot IS NULL
                       AND t.table_name = ? AND t.end_snapshot IS NULL
                       AND c.end_snapshot IS NULL AND LOWER(c.column_type) IN ({}))",
                crate::types::sql_uninlinable_types()
            ),
            params![schema_name, table_name],
            |row| row.get(0),
        )?;
        Ok(!refused)
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

        let mut conn = self.connection();
        let tx = conn.transaction()?;
        enforce_inline_partition_fence(&tx, table_id, base_snapshot)?;
        let snapshot_id =
            finalize_snapshot(&tx, table_id, columns, column_ids, mode, base_snapshot)?;
        if mode != WriteMode::Replace
            && let Some(expected_base_snapshot_id) = expected_base_snapshot_id
        {
            detect_replace_conflict(
                &tx,
                table_id,
                expected_base_snapshot_id,
                ConflictScope::Table,
            )?;
        }
        let schema = batches[0].schema();
        let fields = columns
            .iter()
            .zip(schema.fields())
            .map(|(column, field)| {
                (
                    column.name().to_string(),
                    field.data_type().clone(),
                    column.ducklake_type().to_string(),
                )
            })
            .collect::<Vec<_>>();
        let physical_name = inlined_insert_table(&tx, table_id, snapshot_id, &fields)?;
        let system = inlined_table_layout(&tx, &physical_name)?.system;
        let stats_initialized = table_stats_initialized(&tx, table_id)?;
        seed_stats_if_missing(&tx, table_id)?;
        let mut row_id: i64 = tx.query_row(
            "SELECT next_row_id FROM ducklake_table_stats WHERE table_id = ?",
            params![table_id],
            |row| row.get(0),
        )?;

        let column_list = columns
            .iter()
            .map(|column| quote_ident(column.name()))
            .collect::<Vec<_>>()
            .join(", ");
        for batch in batches {
            for batch_row in 0..batch.num_rows() {
                let encoded = batch
                    .columns()
                    .iter()
                    .zip(columns)
                    .map(|(array, column)| {
                        encode_duckdb_value(array.as_ref(), batch_row, column.ducklake_type(), 0)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let value_list = encoded
                    .iter()
                    .map(|value| value.sql.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                let insert_sql = format!(
                    "INSERT INTO {} ({}, {}, {}, {}) \
                     VALUES (?, ?, ?, {})",
                    quote_ident(&physical_name),
                    system.row_id,
                    system.begin_snapshot,
                    system.end_snapshot,
                    column_list,
                    value_list
                );
                let mut values = Vec::new();
                values.push(Value::BigInt(row_id));
                values.push(Value::BigInt(snapshot_id));
                values.push(Value::Null);
                for value in encoded {
                    values.extend(value.binds);
                }
                tx.execute(&insert_sql, params_from_iter(values))?;
                row_id += 1;
            }
        }

        merge_inlined_table_column_stats(
            &tx,
            table_id,
            columns,
            column_ids,
            batches,
            &[],
            snapshot_id,
            stats_initialized,
        )?;
        let record_count = i64::try_from(record_count).map_err(|_| {
            crate::DuckLakeError::InvalidConfig(
                "register_inlined_data: record count exceeds i64".to_string(),
            )
        })?;
        tx.execute(
            "UPDATE ducklake_table_stats
             SET next_row_id = next_row_id + ?, record_count = record_count + ?
             WHERE table_id = ?",
            params![record_count, record_count, table_id],
        )?;
        // The same composed ledger recording the Parquet path uses: DDL entries
        // plus the write change, appended rather than clobbered.
        record_table_write_changes(
            &tx,
            snapshot_id,
            table_id,
            schema_name,
            table_name,
            mode,
            false,
            true,
            commit_metadata,
        )?;
        let schema_id: i64 = tx.query_row(
            "SELECT schema_id FROM ducklake_table WHERE table_id = ?",
            params![table_id],
            |row| row.get(0),
        )?;
        tx.commit()?;
        Ok(CommitIds {
            snapshot_id,
            schema_id,
            table_id,
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

        for write in writes {
            validate_row_mutation_stage(write)?;
        }
        let mut connection = self.connection();
        let tx = connection.transaction()?;
        if let Some(expected) = expected_base_snapshot_id {
            for write in writes {
                detect_replace_conflict(&tx, write.table_id, expected, ConflictScope::of(write))?;
            }
        }
        for write in writes {
            // Official checks the table-level conflicts first.
            check_staged_conflicts(&tx, write, i64::MAX)?;
            detect_new_inlined_deletes(
                &tx,
                write.table_id,
                write.base_snapshot_id,
                &row_mutation_source_files(write),
            )?;
        }
        let had_live_data = writes
            .iter()
            .map(|write| has_live_data(&tx, write.table_id))
            .collect::<Result<Vec<_>>>()?;
        let (snapshot_id, _schema_version) = insert_snapshot(&tx)?;
        for write in writes {
            if let Some(dated) = rewritten_deletes_snapshot(write) {
                let between: bool = tx.query_row(
                    "SELECT EXISTS(SELECT 1 FROM ducklake_snapshot
                                   WHERE snapshot_id >= ? AND snapshot_id < ?)",
                    params![dated, snapshot_id],
                    |row| row.get(0),
                )?;
                if between {
                    return Err(rewritten_delete_conflict(write.table_id, dated));
                }
            }
        }
        let mut tables = Vec::with_capacity(writes.len());
        for write in writes {
            let schema_id = validate_staged_table(&tx, write)?;
            finalize_table_snapshot(
                &tx,
                snapshot_id,
                write.table_id,
                &write.columns,
                &write.column_ids,
                write.mode,
                write.base_snapshot_id,
                !matches!(write.data, StagedTableData::None),
            )?;
            tables.push(CommitIds {
                snapshot_id,
                schema_id,
                table_id: write.table_id,
            });
        }
        for (write, (had_files, had_inlined_rows)) in writes.iter().zip(had_live_data) {
            match &write.data {
                StagedTableData::Files(files) => {
                    commit_staged_files(&tx, snapshot_id, write, files)?;
                },
                StagedTableData::Inlined(batches) => {
                    commit_staged_inline(&tx, snapshot_id, write, batches)?;
                },
                StagedTableData::None => {},
            }
            for entry in &write.positional_deletes {
                apply_delete_entry(
                    &tx,
                    write.table_id,
                    write.base_snapshot_id,
                    snapshot_id,
                    entry,
                )?;
            }
            replace_delete_files(&tx, write.table_id, &write.rewritten_deletes, || {
                crate::DuckLakeError::Conflict(format!(
                    "a data file of table {} or its delete file changed since snapshot {}; \
                     re-open the catalog and retry the statement",
                    write.table_id, write.base_snapshot_id
                ))
            })?;
            apply_staged_inlined_deletes(&tx, snapshot_id, write)?;
            apply_inlined_file_deletes(
                &tx,
                write.table_id,
                snapshot_id,
                write.base_snapshot_id,
                &write.inlined_file_deletes,
            )?;
            drop_data_files(
                &tx,
                write.table_id,
                snapshot_id,
                write.base_snapshot_id,
                &write.dropped_files,
            )?;
            let changes_made = staged_table_write_changes(write, had_files, had_inlined_rows);
            record_table_changes(
                &tx,
                snapshot_id,
                write.table_id,
                &write.schema_name,
                &write.table_name,
                &changes_made,
                commit_metadata,
            )?;
        }
        tx.commit()?;
        Ok(MultiTableCommit {
            snapshot_id,
            tables,
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
        if rows.iter().collect::<std::collections::HashSet<_>>().len() != rows.len() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "commit_inlined_deletes contains duplicate rows".to_string(),
            ));
        }
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        let (snapshot_id, _schema_version) = insert_snapshot(&tx)?;
        let ours = TableChanges {
            deleted_inlined: true,
            ..TableChanges::default()
        };
        check_table_conflicts(&tx, table_id, base_snapshot, snapshot_id, ours)?;
        let registered = inlined_tables(&tx, table_id)?
            .into_iter()
            .collect::<std::collections::HashMap<_, _>>();
        for row in rows {
            let Some(system) = registered.get(&row.table_name) else {
                return Err(crate::DuckLakeError::Conflict(format!(
                    "inlined row {} belongs to an unregistered table '{}'",
                    row.row_id, row.table_name
                )));
            };
            let affected = tx.execute(
                &format!(
                    "UPDATE {} SET {end} = ? \
                     WHERE {row_id} = ? AND {begin} <= ? AND {end} IS NULL",
                    quote_ident(&row.table_name),
                    end = system.end_snapshot,
                    row_id = system.row_id,
                    begin = system.begin_snapshot,
                ),
                params![snapshot_id, row.row_id, base_snapshot],
            )?;
            if affected != 1 {
                return Err(crate::DuckLakeError::Conflict(format!(
                    "inlined row {} in '{}' is no longer live at snapshot {base_snapshot}",
                    row.row_id, row.table_name
                )));
            }
        }
        let changes_made = crate::metadata_writer::table_storage_changes(
            table_id,
            None,
            ours.deleted_from,
            ours.deleted_inlined,
        );
        record_snapshot_changes(
            &tx,
            snapshot_id,
            &changes_made,
            &SnapshotCommitMetadata::default(),
        )?;
        let schema_id: i64 = tx.query_row(
            "SELECT schema_id FROM ducklake_table WHERE table_id = ?",
            params![table_id],
            |row| row.get(0),
        )?;
        tx.commit()?;
        Ok(CommitIds {
            snapshot_id,
            schema_id,
            table_id,
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
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        let (snapshot_id, _schema_version) = insert_snapshot(&tx)?;
        // Official's table-level check: a drop, alter, compaction or insert of
        // the table since `base_snapshot` refuses the delete.
        let ours = TableChanges {
            deleted_from: true,
            ..TableChanges::default()
        };
        check_table_conflicts(&tx, table_id, base_snapshot, snapshot_id, ours)?;
        apply_delete_entry(
            &tx,
            table_id,
            base_snapshot,
            snapshot_id,
            &DeleteFileEntry {
                data_file_id,
                expected_prev_delete_file,
                delete: delete.clone(),
            },
        )?;
        record_snapshot_changes(
            &tx,
            snapshot_id,
            &format!("deleted_from_table:{table_id}"),
            &SnapshotCommitMetadata::default(),
        )?;
        let schema_id = tx.query_row(
            "SELECT schema_id FROM ducklake_table WHERE table_id = ?",
            params![table_id],
            |row| row.get(0),
        )?;
        tx.commit()?;
        Ok(CommitIds {
            snapshot_id,
            schema_id,
            table_id,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn register_data_file_with_deletes(
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
    ) -> Result<CommitIds> {
        self.register_files_with_deletes(
            table_id,
            schema_name,
            table_name,
            std::slice::from_ref(file),
            deletes,
            mode,
            base_snapshot,
            columns,
            column_ids,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn register_data_files_with_deletes(
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
    ) -> Result<CommitIds> {
        if files.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "register_data_files_with_deletes: files must be non-empty".to_string(),
            ));
        }
        self.register_files_with_deletes(
            table_id,
            schema_name,
            table_name,
            files,
            deletes,
            mode,
            base_snapshot,
            columns,
            column_ids,
        )
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
        validate_delete_entries(WriteMode::Append, deletes)?;
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        let (snapshot_id, _schema_version) = insert_snapshot(&tx)?;
        let ours = TableChanges {
            deleted_from: true,
            ..TableChanges::default()
        };
        check_table_conflicts(&tx, table_id, base_snapshot, snapshot_id, ours)?;
        let sources = SourceFile::of_entries(deletes, true);
        detect_new_inlined_deletes(&tx, table_id, base_snapshot, &sources)?;
        for entry in deletes {
            apply_delete_entry(&tx, table_id, base_snapshot, snapshot_id, entry)?;
        }
        record_snapshot_changes(
            &tx,
            snapshot_id,
            &format!("deleted_from_table:{table_id}"),
            &SnapshotCommitMetadata::default(),
        )?;
        let schema_id = tx.query_row(
            "SELECT schema_id FROM ducklake_table WHERE table_id = ?",
            params![table_id],
            |row| row.get(0),
        )?;
        tx.commit()?;
        Ok(CommitIds {
            snapshot_id,
            schema_id,
            table_id,
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
        // Mixed positional + inlined DELETE in ONE snapshot/transaction,
        // mirroring the SQLite writer's override (the trait default refuses the
        // combination). Single-form deletes keep their dedicated paths.
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
        if inlined
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            != inlined.len()
        {
            return Err(crate::DuckLakeError::InvalidConfig(
                "commit_deletes contains duplicate inlined rows".to_string(),
            ));
        }
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        let (snapshot_id, _schema_version) = insert_snapshot(&tx)?;
        let ours = TableChanges {
            deleted_from: !positional.is_empty(),
            deleted_inlined: !inlined.is_empty(),
            ..TableChanges::default()
        };
        check_table_conflicts(&tx, table_id, base_snapshot, snapshot_id, ours)?;
        let sources = SourceFile::of_entries(positional, true);
        detect_new_inlined_deletes(&tx, table_id, base_snapshot, &sources)?;
        for entry in positional {
            apply_delete_entry(&tx, table_id, base_snapshot, snapshot_id, entry)?;
        }
        apply_inlined_deletes(&tx, table_id, snapshot_id, base_snapshot, inlined)?;
        record_snapshot_changes(
            &tx,
            snapshot_id,
            &crate::metadata_writer::table_storage_changes(
                table_id,
                None,
                ours.deleted_from,
                ours.deleted_inlined,
            ),
            &SnapshotCommitMetadata::default(),
        )?;
        let schema_id: i64 = tx.query_row(
            "SELECT schema_id FROM ducklake_table WHERE table_id = ?",
            params![table_id],
            |row| row.get(0),
        )?;
        tx.commit()?;
        Ok(CommitIds {
            snapshot_id,
            schema_id,
            table_id,
        })
    }

    fn commit_compaction(
        &self,
        table_id: i64,
        base_snapshot: i64,
        sources: &[CompactionSourceFile],
        outputs: &[CompactionOutputFile],
        retirement: SourceRetirement,
    ) -> Result<CommitIds> {
        if sources.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "commit_compaction requires at least one source file".to_string(),
            ));
        }
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        let (snapshot_id, _schema_version) = insert_snapshot(&tx)?;
        let source_files = sources
            .iter()
            .map(|source| SourceFile::any(source.data_file_id))
            .collect::<Vec<_>>();
        detect_new_inlined_deletes(&tx, table_id, base_snapshot, &source_files)?;
        for source in sources {
            let live: Option<i64> = tx
                .query_row(
                    "SELECT 1 FROM ducklake_data_file
                     WHERE data_file_id = ? AND table_id = ? AND end_snapshot IS NULL",
                    params![source.data_file_id, table_id],
                    |row| row.get(0),
                )
                .optional()?;
            if live.is_none() {
                return Err(crate::DuckLakeError::Conflict(format!(
                    "compaction of table {table_id} could not commit: source data file {} is no \
                     longer live since snapshot {base_snapshot}; re-open the catalog and re-plan",
                    source.data_file_id
                )));
            }
            let current_delete: Option<i64> = tx
                .query_row(
                    "SELECT delete_file_id FROM ducklake_delete_file
                     WHERE data_file_id = ? AND end_snapshot IS NULL",
                    params![source.data_file_id],
                    |row| row.get(0),
                )
                .optional()?;
            if current_delete != source.delete_file_id {
                return Err(crate::DuckLakeError::Conflict(format!(
                    "compaction of table {table_id} could not commit: the live delete file of \
                     source data file {} changed from {:?} to {current_delete:?} since snapshot \
                     {base_snapshot}; re-open the catalog and re-plan",
                    source.data_file_id, source.delete_file_id
                )));
            }
        }

        let source_ids = sources
            .iter()
            .map(|source| source.data_file_id)
            .collect::<Vec<_>>();
        let source_ids = id_list(&source_ids);
        match retirement {
            SourceRetirement::Remove => {
                tx.execute_batch(&format!(
                    "INSERT INTO ducklake_files_scheduled_for_deletion
                         (data_file_id, path, path_is_relative)
                     SELECT df.data_file_id, {RESOLVED_PATH}, {REL_FLAG}
                     FROM ducklake_data_file df
                     JOIN ducklake_table t ON t.table_id = df.table_id
                     JOIN ducklake_schema s ON s.schema_id = t.schema_id
                     WHERE df.data_file_id IN ({source_ids});
                     INSERT INTO ducklake_files_scheduled_for_deletion
                         (data_file_id, path, path_is_relative)
                     SELECT df.delete_file_id, {RESOLVED_PATH}, {REL_FLAG}
                     FROM ducklake_delete_file df
                     JOIN ducklake_table t ON t.table_id = df.table_id
                     JOIN ducklake_schema s ON s.schema_id = t.schema_id
                     WHERE df.data_file_id IN ({source_ids});
                     DELETE FROM ducklake_delete_file WHERE data_file_id IN ({source_ids});
                     DELETE FROM ducklake_data_file WHERE data_file_id IN ({source_ids});
                     DELETE FROM ducklake_file_column_stats WHERE data_file_id IN ({source_ids});
                     DELETE FROM ducklake_file_partition_value
                     WHERE data_file_id IN ({source_ids})"
                ))?;
            },
            SourceRetirement::Retire => {
                tx.execute(
                    &format!(
                        "UPDATE ducklake_data_file SET end_snapshot = ?
                         WHERE data_file_id IN ({source_ids}) AND end_snapshot IS NULL"
                    ),
                    params![snapshot_id],
                )?;
                tx.execute(
                    &format!(
                        "UPDATE ducklake_delete_file SET end_snapshot = ?
                         WHERE data_file_id IN ({source_ids}) AND end_snapshot IS NULL"
                    ),
                    params![snapshot_id],
                )?;
            },
        }

        let output_count = i64::try_from(outputs.len()).map_err(|_| {
            crate::DuckLakeError::InvalidConfig(
                "commit_compaction output count exceeds i64".to_string(),
            )
        })?;
        let output_ids = reserve_file_ids(&tx, output_count)?;
        for (output, data_file_id) in outputs.iter().zip(output_ids) {
            let begin_snapshot = output.begin_snapshot.unwrap_or(snapshot_id);
            tx.execute(
                "INSERT INTO ducklake_data_file
                     (data_file_id, table_id, path, path_is_relative, file_size_bytes,
                      footer_size, record_count, row_id_start, begin_snapshot, partial_max)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params![
                    data_file_id,
                    table_id,
                    output.file.path.as_str(),
                    output.file.path_is_relative,
                    output.file.file_size_bytes,
                    output.file.footer_size,
                    output.file.record_count,
                    crate::metadata_writer::carried_row_id_start(&output.file),
                    begin_snapshot,
                    output.partial_max,
                ],
            )?;
            insert_file_column_stats(&tx, table_id, data_file_id, &output.file.column_stats)?;
            insert_partition_metadata(&tx, table_id, data_file_id, &output.file)?;
        }
        // `ducklake_table_stats` is left as it is, as official's compaction
        // leaves it: `record_count` stays the gross count of rows ever inserted.
        record_snapshot_changes(
            &tx,
            snapshot_id,
            &crate::metadata_writer::compaction_change(retirement, table_id),
            &SnapshotCommitMetadata::new(),
        )?;
        let schema_id = tx.query_row(
            "SELECT schema_id FROM ducklake_table WHERE table_id = ?",
            params![table_id],
            |row| row.get(0),
        )?;
        tx.commit()?;
        Ok(CommitIds {
            snapshot_id,
            schema_id,
            table_id,
        })
    }

    fn commit_inlined_flush(
        &self,
        base_snapshot: i64,
        tables: &[InlinedFlushTable],
    ) -> Result<i64> {
        validate_inlined_flush(tables)?;
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        // A flush is not DDL: schema_version carries forward.
        let (snapshot_id, _schema_version) = insert_snapshot(&tx)?;
        let changes = {
            let mut statement = tx.prepare(
                "SELECT changes_made FROM ducklake_snapshot_changes
                 WHERE snapshot_id > ? AND snapshot_id < ?",
            )?;
            statement
                .query_map(params![base_snapshot, snapshot_id], |row| {
                    row.get::<_, Option<String>>(0)
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        if let Some(conflict) =
            crate::metadata_writer::inlined_flush_table_conflict(tables, &changes)
        {
            return Err(conflict);
        }
        for table in tables {
            let live: Option<i64> = tx
                .query_row(
                    "SELECT 1 FROM ducklake_table WHERE table_id = ? AND end_snapshot IS NULL",
                    params![table.table_id],
                    |row| row.get(0),
                )
                .optional()?;
            if live.is_none() {
                return Err(inlined_flush_conflict(table.table_id, base_snapshot));
            }
            seed_stats_if_missing(&tx, table.table_id)?;
            let output_count = i64::try_from(table.files.len()).map_err(|_| {
                crate::DuckLakeError::InvalidConfig("flush file count exceeds i64".to_string())
            })?;
            let output_ids = reserve_file_ids(&tx, output_count)?;
            let mut added_bytes = 0;
            for (out, data_file_id) in table.files.iter().zip(output_ids) {
                tx.execute(
                    "INSERT INTO ducklake_data_file
                         (data_file_id, table_id, path, path_is_relative, file_size_bytes,
                          footer_size, record_count, row_id_start, begin_snapshot, partial_max)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    params![
                        data_file_id,
                        table.table_id,
                        out.file.path.as_str(),
                        out.file.path_is_relative,
                        out.file.file_size_bytes,
                        out.file.footer_size,
                        out.file.record_count,
                        carried_row_id_start(&out.file),
                        out.begin_snapshot,
                        out.partial_max,
                    ],
                )?;
                insert_file_column_stats(
                    &tx,
                    table.table_id,
                    data_file_id,
                    &out.file.column_stats,
                )?;
                insert_partition_metadata(&tx, table.table_id, data_file_id, &out.file)?;
                if let Some(delete) = &out.delete {
                    insert_flush_delete_file(&tx, table.table_id, data_file_id, delete)?;
                }
                added_bytes += out.file.file_size_bytes;
            }
            replace_delete_files(&tx, table.table_id, &table.file_deletes, || {
                inlined_flush_conflict(table.table_id, base_snapshot)
            })?;
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
                tx.execute(
                    &format!(
                        "DELETE FROM {} WHERE file_id IN ({ids}) AND begin_snapshot <= ?",
                        quote_ident(&inlined_delete)
                    ),
                    params![base_snapshot],
                )?;
            }
            for name in &table.inlined_tables {
                let system = inlined_table_layout(&tx, name)?.system;
                tx.execute(
                    &format!(
                        "DELETE FROM {} WHERE {} <= ?",
                        quote_ident(name),
                        system.begin_snapshot,
                    ),
                    params![base_snapshot],
                )?;
            }
            if !table.files.is_empty() {
                tx.execute(
                    "UPDATE ducklake_table_stats SET file_size_bytes = file_size_bytes + ?
                     WHERE table_id = ?",
                    params![added_bytes, table.table_id],
                )?;
                let fields = live_stat_fields(&tx, table.table_id)?;
                merge_file_table_column_stats(
                    &tx,
                    table.table_id,
                    &fields,
                    &crate::metadata_writer::file_column_stats(
                        table.files.iter().map(|flushed| &flushed.file),
                    ),
                )?;
            }
        }
        record_snapshot_changes(
            &tx,
            snapshot_id,
            &inlined_flush_changes(tables),
            &SnapshotCommitMetadata::default(),
        )?;
        tx.commit()?;
        Ok(snapshot_id)
    }

    fn retire_appends_since(&self, table_id: i64, base_snapshot: i64) -> Result<Option<i64>> {
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        let (snapshot_id, _schema_version) = insert_snapshot(&tx)?;
        // A delete file rewritten after base begins at its earliest deletion,
        // which may precede base; its `partial_max` is its latest one.
        let impure_delete: Option<i64> = tx
            .query_row(
                "SELECT 1 FROM ducklake_delete_file
                 WHERE table_id = ? AND (begin_snapshot > ? OR partial_max > ?) LIMIT 1",
                params![table_id, base_snapshot, base_snapshot],
                |row| row.get(0),
            )
            .optional()?;
        // A base-era file ended since, or a live file holding rows of both sides
        // of base (a flush's file begins at its earliest row).
        let impure_ended: Option<i64> = tx
            .query_row(
                "SELECT 1 FROM ducklake_data_file
                 WHERE table_id = ? AND begin_snapshot <= ?
                   AND ((end_snapshot IS NOT NULL AND end_snapshot > ?)
                        OR (end_snapshot IS NULL AND partial_max > ?)) LIMIT 1",
                params![table_id, base_snapshot, base_snapshot, base_snapshot],
                |row| row.get(0),
            )
            .optional()?;
        let impure_column: Option<i64> = tx
            .query_row(
                "SELECT 1 FROM ducklake_column
                 WHERE table_id = ? AND (begin_snapshot > ? OR end_snapshot > ?) LIMIT 1",
                params![table_id, base_snapshot, base_snapshot],
                |row| row.get(0),
            )
            .optional()?;
        let impure_partition: Option<i64> = tx
            .query_row(
                "SELECT 1 FROM ducklake_partition_info
                 WHERE table_id = ? AND (begin_snapshot > ? OR end_snapshot > ?) LIMIT 1",
                params![table_id, base_snapshot, base_snapshot],
                |row| row.get(0),
            )
            .optional()?;
        // Inlined rows: a row ended after base is a deletion (or an update's old
        // version), as is any inlined deletion of a Parquet row.
        let table_exists = |name: &str| -> Result<bool> {
            Ok(tx.query_row(
                "SELECT COUNT(*) > 0 FROM information_schema.tables WHERE table_name = ?",
                params![name],
                |row| row.get(0),
            )?)
        };
        let inlined = if table_exists("ducklake_inlined_data_tables")? {
            inlined_tables(&tx, table_id)?
        } else {
            Vec::new()
        };
        let mut impure_inlined = false;
        for (name, system) in &inlined {
            let ended: Option<i64> = tx
                .query_row(
                    &format!(
                        "SELECT 1 FROM {} WHERE {} > ? LIMIT 1",
                        quote_ident(name),
                        system.end_snapshot
                    ),
                    params![base_snapshot],
                    |row| row.get(0),
                )
                .optional()?;
            impure_inlined |= ended.is_some();
        }
        let inlined_deletes = crate::metadata_provider::inlined_delete_table_name(table_id)?;
        if table_exists(&inlined_deletes)? {
            let deleted: Option<i64> = tx
                .query_row(
                    &format!(
                        "SELECT 1 FROM {} WHERE begin_snapshot > ? LIMIT 1",
                        quote_ident(&inlined_deletes)
                    ),
                    params![base_snapshot],
                    |row| row.get(0),
                )
                .optional()?;
            impure_inlined |= deleted.is_some();
        }
        if impure_delete.is_some()
            || impure_ended.is_some()
            || impure_column.is_some()
            || impure_partition.is_some()
            || impure_inlined
        {
            return Err(crate::DuckLakeError::Conflict(format!(
                "table {table_id}: changes since snapshot {base_snapshot} are not a pure append \
                 (delete/replace/update or schema/partition change present); refusing to retire"
            )));
        }
        let appended_rows: Option<i64> = tx.query_row(
            "SELECT CAST(SUM(record_count) AS BIGINT) FROM ducklake_data_file
             WHERE table_id = ? AND end_snapshot IS NULL AND begin_snapshot > ?",
            params![table_id, base_snapshot],
            |row| row.get(0),
        )?;
        let mut appended_inlined = 0_i64;
        for (name, system) in &inlined {
            appended_inlined += tx.query_row(
                &format!(
                    "SELECT COUNT(*) FROM {} WHERE {} > ? AND {} IS NULL",
                    quote_ident(name),
                    system.begin_snapshot,
                    system.end_snapshot
                ),
                params![base_snapshot],
                |row| row.get::<_, i64>(0),
            )?;
        }
        if appended_rows.is_none() && appended_inlined == 0 {
            return Ok(None);
        }
        tx.execute(
            "UPDATE ducklake_data_file SET end_snapshot = ?
             WHERE table_id = ? AND end_snapshot IS NULL AND begin_snapshot > ?",
            params![snapshot_id, table_id, base_snapshot],
        )?;
        for (name, system) in &inlined {
            tx.execute(
                &format!(
                    "UPDATE {} SET {} = ? WHERE {} > ? AND {} IS NULL",
                    quote_ident(name),
                    system.end_snapshot,
                    system.begin_snapshot,
                    system.end_snapshot
                ),
                params![snapshot_id, base_snapshot],
            )?;
        }
        // `ducklake_table_stats` is left as it is, as official's `DELETE`
        // leaves it: `record_count` stays the gross count of rows ever inserted.
        let changes_made = match (appended_rows.is_some(), appended_inlined > 0) {
            (true, true) => format!("deleted_from_table:{table_id},inlined_delete:{table_id}"),
            (false, true) => format!("inlined_delete:{table_id}"),
            _ => format!("deleted_from_table:{table_id}"),
        };
        record_snapshot_changes(
            &tx,
            snapshot_id,
            &changes_made,
            &SnapshotCommitMetadata::default(),
        )?;
        tx.commit()?;
        Ok(Some(snapshot_id))
    }

    fn commit_truncate(
        &self,
        table_id: i64,
        _schema_name: &str,
        _table_name: &str,
        base_snapshot: i64,
    ) -> Result<u64> {
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        let (snapshot_id, _schema_version) = insert_snapshot(&tx)?;
        let has_live_data: Option<i64> = tx
            .query_row(
                "SELECT 1 FROM ducklake_data_file
                 WHERE table_id = ? AND end_snapshot IS NULL LIMIT 1",
                params![table_id],
                |row| row.get(0),
            )
            .optional()?;
        let live_inlined = live_inlined_row_count(&tx, table_id)?;
        if has_live_data.is_none() && live_inlined == 0 {
            return Ok(0);
        }
        let ours = TableChanges {
            deleted_from: has_live_data.is_some(),
            deleted_inlined: live_inlined > 0,
            ..TableChanges::default()
        };
        check_table_conflicts(&tx, table_id, base_snapshot, snapshot_id, ours)?;
        let gross: Option<i64> = tx
            .query_row(
                "SELECT COALESCE(SUM(record_count), 0) FROM ducklake_data_file WHERE table_id = ? AND end_snapshot IS NULL",
                params![table_id],
                |row| row.get(0),
            )
            .optional()?;
        let deleted = tx.query_row(
            "SELECT COALESCE(SUM(delete_count), 0) FROM ducklake_delete_file
             WHERE table_id = ? AND end_snapshot IS NULL",
            params![table_id],
            |row| row.get::<_, i64>(0),
        )?;
        let live_rows =
            u64::try_from((gross.unwrap_or(0) + live_inlined - deleted).max(0)).unwrap_or(0);
        tx.execute(
            "UPDATE ducklake_data_file SET end_snapshot = ?
             WHERE table_id = ? AND end_snapshot IS NULL",
            params![snapshot_id, table_id],
        )?;
        for (table_name, system) in inlined_tables(&tx, table_id)? {
            tx.execute(
                &format!(
                    "UPDATE {} SET {end} = ? WHERE {end} IS NULL",
                    quote_ident(&table_name),
                    end = system.end_snapshot,
                ),
                params![snapshot_id],
            )?;
        }
        tx.execute(
            "UPDATE ducklake_delete_file SET end_snapshot = ?
             WHERE table_id = ? AND end_snapshot IS NULL",
            params![snapshot_id, table_id],
        )?;
        record_snapshot_changes(
            &tx,
            snapshot_id,
            &crate::metadata_writer::table_storage_changes(
                table_id,
                None,
                ours.deleted_from,
                ours.deleted_inlined,
            ),
            &SnapshotCommitMetadata::default(),
        )?;
        tx.commit()?;
        Ok(live_rows)
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
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        let partition_id: i64 =
            tx.query_row("SELECT nextval('ducklake_partition_id_seq')", [], |row| {
                row.get(0)
            })?;
        let (new_snapshot, _carried) = insert_snapshot(&tx)?;

        // Resolve each partition-key column NAME to its live column_id.
        let mut column_ids: Vec<i64> = Vec::with_capacity(columns.len());
        for (name, _transform) in columns {
            let column_id: i64 = tx
                .query_row(
                    "SELECT column_id FROM ducklake_column
                     WHERE table_id = ? AND column_name = ? AND end_snapshot IS NULL
                       AND parent_column IS NULL",
                    params![table_id, name.as_str()],
                    |row| row.get(0),
                )
                .optional()?
                .ok_or_else(|| {
                    crate::DuckLakeError::InvalidConfig(format!(
                        "set_partition_spec: no live column '{name}' in table {table_id}"
                    ))
                })?;
            column_ids.push(column_id);
        }

        tx.execute(
            "UPDATE ducklake_partition_info SET end_snapshot = ?
             WHERE table_id = ? AND end_snapshot IS NULL",
            params![new_snapshot, table_id],
        )?;
        tx.execute(
            "INSERT INTO ducklake_partition_info
                 (partition_id, table_id, begin_snapshot, end_snapshot)
             VALUES (?, ?, ?, NULL)",
            params![partition_id, table_id, new_snapshot],
        )?;
        for (key_index, column_id) in column_ids.iter().enumerate() {
            tx.execute(
                "INSERT INTO ducklake_partition_column
                     (partition_id, table_id, partition_key_index, column_id, transform)
                 VALUES (?, ?, ?, ?, ?)",
                params![
                    partition_id,
                    table_id,
                    key_index as i64,
                    *column_id,
                    columns[key_index].1.to_catalog_string()
                ],
            )?;
        }

        let new_schema_version = bump_schema_version(&tx, new_snapshot)?;
        record_schema_version(&tx, new_snapshot, new_schema_version, table_id)?;
        create_eager_inlined_table(&tx, table_id, new_schema_version)?;
        record_snapshot_changes(
            &tx,
            new_snapshot,
            &format!("altered_table:{table_id}"),
            &SnapshotCommitMetadata::default(),
        )?;
        tx.commit()?;
        Ok(new_snapshot)
    }

    fn live_partition_spec(
        &self,
        table_id: i64,
    ) -> Result<Option<crate::partition::PartitionSpec>> {
        let conn = self.connection();
        let mut stmt = conn.prepare(
            "SELECT pi.partition_id, pc.partition_key_index, pc.column_id, pc.transform
             FROM ducklake_partition_info AS pi
             JOIN ducklake_partition_column AS pc
               ON pc.partition_id = pi.partition_id AND pc.table_id = pi.table_id
             WHERE pi.table_id = ? AND pi.end_snapshot IS NULL
             ORDER BY pc.partition_key_index",
        )?;
        let rows = stmt
            .query_map(duckdb::params![table_id], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    i32::try_from(row.get::<_, i64>(1)?).unwrap_or(0),
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        // prune_safe = false: this spec is for laying out a write, never pruning.
        Ok(crate::partition::PartitionSpec::from_rows(rows, false))
    }

    fn reset_partition_spec(&self, table_id: i64) -> Result<i64> {
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        let (new_snapshot, _carried) = insert_snapshot(&tx)?;
        let ended = tx.execute(
            "UPDATE ducklake_partition_info SET end_snapshot = ?
             WHERE table_id = ? AND end_snapshot IS NULL",
            params![new_snapshot, table_id],
        )?;
        if ended == 0 {
            // Nothing to reset: roll back the snapshot and report the head.
            drop(tx);
            let head: i64 = conn.query_row(
                "SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot",
                [],
                |row| row.get(0),
            )?;
            return Ok(head);
        }
        let new_schema_version = bump_schema_version(&tx, new_snapshot)?;
        record_schema_version(&tx, new_snapshot, new_schema_version, table_id)?;
        create_eager_inlined_table(&tx, table_id, new_schema_version)?;
        record_snapshot_changes(
            &tx,
            new_snapshot,
            &format!("altered_table:{table_id}"),
            &SnapshotCommitMetadata::default(),
        )?;
        tx.commit()?;
        Ok(new_snapshot)
    }

    fn live_sort_spec(&self, table_id: i64) -> Result<Option<crate::sort::SortSpec>> {
        let conn = self.connection();
        let mut stmt = conn.prepare(
            "SELECT si.sort_id, se.sort_key_index, se.expression, se.dialect,
                    se.sort_direction, se.null_order
             FROM ducklake_sort_info AS si
             JOIN ducklake_sort_expression AS se
               ON se.sort_id = si.sort_id AND se.table_id = si.table_id
             WHERE si.table_id = ? AND si.end_snapshot IS NULL
             ORDER BY se.sort_key_index",
        )?;
        let rows = stmt
            .query_map(duckdb::params![table_id], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    i32::try_from(row.get::<_, i64>(1)?).unwrap_or(0),
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(crate::sort::SortSpec::from_rows(rows))
    }

    fn set_sort_spec(&self, table_id: i64, fields: &[crate::sort::SortField]) -> Result<i64> {
        if fields.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "set_sort_spec: at least one sort key is required (use reset_sort_spec to clear)"
                    .to_string(),
            ));
        }
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        let sort_id: i64 = tx.query_row("SELECT nextval('ducklake_sort_id_seq')", [], |row| {
            row.get(0)
        })?;
        let (new_snapshot, _carried) = insert_snapshot(&tx)?;

        // Validate every sort key resolves to a live column (v1 supports bare column
        // keys only), so a bad SET fails here rather than silently producing
        // unsorted writes later.
        for field in fields {
            let column = field.column_candidate().ok_or_else(|| {
                crate::DuckLakeError::InvalidConfig(format!(
                    "set_sort_spec: sort key '{}' is not a bare column; only column \
                     sort keys are supported",
                    field.expression
                ))
            })?;
            let exists: Option<i64> = tx
                .query_row(
                    "SELECT column_id FROM ducklake_column
                     WHERE table_id = ? AND column_name = ? AND end_snapshot IS NULL
                       AND parent_column IS NULL",
                    params![table_id, column.as_str()],
                    |row| row.get(0),
                )
                .optional()?;
            if exists.is_none() {
                return Err(crate::DuckLakeError::InvalidConfig(format!(
                    "set_sort_spec: no live column '{column}' in table {table_id}"
                )));
            }
        }

        tx.execute(
            "UPDATE ducklake_sort_info SET end_snapshot = ?
             WHERE table_id = ? AND end_snapshot IS NULL",
            params![new_snapshot, table_id],
        )?;
        tx.execute(
            "INSERT INTO ducklake_sort_info
                 (sort_id, table_id, begin_snapshot, end_snapshot)
             VALUES (?, ?, ?, NULL)",
            params![sort_id, table_id, new_snapshot],
        )?;
        for field in fields {
            tx.execute(
                "INSERT INTO ducklake_sort_expression
                     (sort_id, table_id, sort_key_index, expression, dialect,
                      sort_direction, null_order)
                 VALUES (?, ?, ?, ?, ?, ?, ?)",
                params![
                    sort_id,
                    table_id,
                    field.sort_key_index as i64,
                    field.expression.as_str(),
                    field.dialect.as_str(),
                    field.direction.to_catalog_string(),
                    field.null_order.to_catalog_string()
                ],
            )?;
        }

        // A sort-order change does NOT bump schema_version.
        record_snapshot_changes(
            &tx,
            new_snapshot,
            &format!("altered_table:{table_id}"),
            &SnapshotCommitMetadata::default(),
        )?;
        tx.commit()?;
        Ok(new_snapshot)
    }

    fn reset_sort_spec(&self, table_id: i64) -> Result<i64> {
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        let (new_snapshot, _carried) = insert_snapshot(&tx)?;
        let ended = tx.execute(
            "UPDATE ducklake_sort_info SET end_snapshot = ?
             WHERE table_id = ? AND end_snapshot IS NULL",
            params![new_snapshot, table_id],
        )?;
        if ended == 0 {
            // Nothing to reset: roll back the snapshot and report the head.
            drop(tx);
            let head: i64 = conn.query_row(
                "SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot",
                [],
                |row| row.get(0),
            )?;
            return Ok(head);
        }
        // A sort-order change does NOT bump schema_version.
        record_snapshot_changes(
            &tx,
            new_snapshot,
            &format!("altered_table:{table_id}"),
            &SnapshotCommitMetadata::default(),
        )?;
        tx.commit()?;
        Ok(new_snapshot)
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
        // Fileless commit point (CREATE TABLE, zero-row Replace). Single-catalog
        // DuckDB defers the snapshot-row insert out of begin_write_transaction,
        // so this is NOT the trait's no-op default: it inserts the deferred
        // snapshot row + column generation and, for Replace, retires the prior
        // generation — making the new head visible atomically.
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        let snapshot_id =
            finalize_snapshot(&tx, table_id, columns, column_ids, mode, base_snapshot)?;
        record_table_write_changes(
            &tx,
            snapshot_id,
            table_id,
            schema_name,
            table_name,
            mode,
            false,
            false,
            &SnapshotCommitMetadata::default(),
        )?;
        let schema_id: i64 = tx.query_row(
            "SELECT schema_id FROM ducklake_table WHERE table_id = ?",
            params![table_id],
            |row| row.get(0),
        )?;
        tx.commit()?;
        Ok(CommitIds {
            snapshot_id,
            schema_id,
            table_id,
        })
    }

    fn end_table_files(&self, table_id: i64, snapshot_id: i64) -> Result<u64> {
        // Used by WriteMode::Replace. End-snapshotting every visible file drops
        // the table's currently-visible row count and byte total to zero.
        // next_row_id is deliberately NOT reset (rowids stay monotonic).
        let mut conn = self.connection();
        let tx = conn.transaction()?;

        let rows_affected = tx.execute(
            "UPDATE ducklake_data_file SET end_snapshot = ?
             WHERE table_id = ? AND end_snapshot IS NULL",
            params![snapshot_id, table_id],
        )? as u64;

        tx.commit()?;
        Ok(rows_affected)
    }

    fn get_data_path(&self) -> Result<String> {
        let conn = self.connection();
        let row: Option<String> = conn
            .query_row(
                "SELECT value FROM ducklake_metadata WHERE key = ? AND scope IS NULL",
                params!["data_path"],
                |row| row.get(0),
            )
            .optional()?;
        match row {
            Some(path) => Ok(path),
            None => Err(crate::error::DuckLakeError::InvalidConfig(
                "Missing required catalog metadata: 'data_path' not configured.".to_string(),
            )),
        }
    }

    fn get_table_column_nullability(
        &self,
        schema_name: &str,
        table_name: &str,
    ) -> Result<Option<Vec<(String, bool)>>> {
        let conn = self.connection();
        let mut statement = conn.prepare(
            "SELECT c.column_name, c.nulls_allowed
             FROM ducklake_column c
             JOIN ducklake_table t ON t.table_id = c.table_id
             JOIN ducklake_schema s ON s.schema_id = t.schema_id
             WHERE s.schema_name = ? AND s.end_snapshot IS NULL
               AND t.table_name = ? AND t.end_snapshot IS NULL
               AND c.end_snapshot IS NULL AND c.parent_column IS NULL
             ORDER BY c.column_order",
        )?;
        let rows = statement.query_map(params![schema_name, table_name], |row| {
            let name: String = row.get(0)?;
            let nullable: Option<bool> = row.get(1)?;
            Ok((name, nullable.unwrap_or(true)))
        })?;
        let columns = rows.collect::<std::result::Result<Vec<_>, _>>()?;
        Ok((!columns.is_empty()).then_some(columns))
    }

    fn set_data_path(&self, path: &str) -> Result<()> {
        let path = directory_path(path);
        let mut conn = self.connection();
        let current: Option<String> = conn
            .query_row(
                "SELECT value FROM ducklake_metadata WHERE key = 'data_path' AND scope IS NULL",
                [],
                |row| row.get(0),
            )
            .optional()?;
        if current.as_deref() == Some(path.as_str()) {
            return Ok(());
        }

        let tx = conn.transaction()?;
        tx.execute(
            "DELETE FROM ducklake_metadata WHERE key = 'data_path' AND scope IS NULL",
            [],
        )?;
        tx.execute(
            "INSERT INTO ducklake_metadata (key, value, scope) VALUES ('data_path', ?, NULL)",
            params![path],
        )?;
        tx.commit()?;
        Ok(())
    }

    fn initialize_schema(&self) -> Result<()> {
        let conn = self.connection();
        conn.execute_batch(SQL_CREATE_SCHEMA)?;
        conn.execute_batch("ALTER TABLE ducklake_metadata ADD COLUMN IF NOT EXISTS scope VARCHAR")?;
        conn.execute_batch(
            "ALTER TABLE ducklake_metadata ADD COLUMN IF NOT EXISTS scope_id BIGINT",
        )?;
        conn.execute_batch(
            "ALTER TABLE ducklake_snapshot
                 ADD COLUMN IF NOT EXISTS next_catalog_id BIGINT DEFAULT 1;
             ALTER TABLE ducklake_snapshot
                 ADD COLUMN IF NOT EXISTS next_file_id BIGINT DEFAULT 0;
             ALTER TABLE ducklake_schema
                 ADD COLUMN IF NOT EXISTS schema_uuid UUID DEFAULT uuid();
             ALTER TABLE ducklake_table
                 ADD COLUMN IF NOT EXISTS table_uuid UUID DEFAULT uuid();
             ALTER TABLE ducklake_data_file
                 ADD COLUMN IF NOT EXISTS file_order BIGINT;
             ALTER TABLE ducklake_data_file
                 ADD COLUMN IF NOT EXISTS file_format VARCHAR;
             ALTER TABLE ducklake_delete_file
                 ADD COLUMN IF NOT EXISTS format VARCHAR",
        )?;
        // Upgrade a pre-existing catalog to carry ducklake_data_file.partition_id
        // (CREATE TABLE IF NOT EXISTS never alters an existing table). DuckDB supports
        // ADD COLUMN IF NOT EXISTS, so this is idempotent and lossless (NULL means
        // "not partitioned").
        conn.execute_batch(
            "ALTER TABLE ducklake_data_file ADD COLUMN IF NOT EXISTS partition_id BIGINT",
        )?;
        conn.execute_batch(
            "ALTER TABLE ducklake_snapshot_changes ALTER COLUMN changes_made DROP NOT NULL;
             ALTER TABLE ducklake_data_file ADD COLUMN IF NOT EXISTS partial_max BIGINT;
             ALTER TABLE ducklake_delete_file ADD COLUMN IF NOT EXISTS partial_max BIGINT",
        )?;
        // Seed the monotonic column_id allocator (snapshot_id uses MAX+1, and
        // schema/table/data_file/delete_file ids use sequences, so none of those
        // need a counter). Idempotent on re-open, and seeded from the current MAX
        // so a pre-existing catalog continues without reusing ids.
        conn.execute_batch(
            "INSERT INTO ducklake_metadata (key, value, scope)
             SELECT 'next_column_id',
                    CAST(COALESCE((SELECT MAX(column_id) FROM ducklake_column), 0) AS VARCHAR),
                    NULL
             WHERE NOT EXISTS (
                 SELECT 1 FROM ducklake_metadata WHERE key = 'next_column_id' AND scope IS NULL
             )",
        )?;
        // Give legacy directory paths the trailing `/` DuckLake readers expect.
        conn.execute_batch(
            "UPDATE ducklake_metadata SET value = value || '/'
             WHERE key = 'data_path' AND scope IS NULL AND value <> '' AND value NOT LIKE '%/';
             UPDATE ducklake_schema SET path = path || '/' WHERE path <> '' AND path NOT LIKE '%/';
             UPDATE ducklake_table SET path = path || '/' WHERE path <> '' AND path NOT LIKE '%/';",
        )?;
        Ok(())
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
        let mut conn = self.connection();
        let tx = conn.transaction()?;

        // Reserve the column ids first. These match the staged parquet field ids.
        let catalog_columns = catalog_column_defs(columns)?;
        let n = catalog_columns.len() as i64;
        let last_column_id = reserve_ids(&tx, "next_column_id", n)?;
        // Freshly reserved ids. Only a genuinely-new column consumes one below; an
        // existing column keeps its current id, so some may go unused (harmless
        // monotonic-counter gaps).
        let fresh_ids: Vec<i64> = ((last_column_id - n + 1)..=last_column_id).collect();

        // The catalog head this write is based on; a Replace commit aborts if
        // another writer published a newer generation of the table past it.
        let base_snapshot_id: i64 = tx.query_row(
            "SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot",
            [],
            |row| row.get(0),
        )?;

        // Tentative id for WriteSetupResult; the real one is assigned at the
        // commit (finalize_snapshot), so it may differ under concurrency.
        let snapshot_id: i64 = base_snapshot_id + 1;

        let schema_id: i64 = {
            let existing: Option<i64> = tx
                .query_row(
                    "SELECT schema_id FROM ducklake_schema
                     WHERE schema_name = ? AND end_snapshot IS NULL",
                    params![schema_name],
                    |row| row.get(0),
                )
                .optional()?;
            match existing {
                Some(id) => id,
                None => tx.query_row(
                    "INSERT INTO ducklake_schema (schema_name, path, path_is_relative, begin_snapshot)
                     VALUES (?, ?, true, ?) RETURNING schema_id",
                    params![schema_name, directory_path(schema_name), snapshot_id],
                    |row| row.get(0),
                )?,
            }
        };

        let table_id: i64 = {
            let existing: Option<i64> = tx
                .query_row(
                    "SELECT table_id FROM ducklake_table
                     WHERE schema_id = ? AND table_name = ? AND end_snapshot IS NULL",
                    params![schema_id, table_name],
                    |row| row.get(0),
                )
                .optional()?;
            match existing {
                Some(id) => id,
                None => tx.query_row(
                    "INSERT INTO ducklake_table (schema_id, table_name, path, path_is_relative, begin_snapshot)
                     VALUES (?, ?, ?, true, ?) RETURNING table_id",
                    params![schema_id, table_name, directory_path(table_name), snapshot_id],
                    |row| row.get(0),
                )?,
            }
        };

        // Existing columns to (a) check schema compatibility and (b) REUSE each
        // column's id (column_id == parquet field_id == a column's stable
        // identity). Collected into owned Vec so the statement drops before the
        // commit that follows.
        let existing_rows: Vec<(String, String, i64, Option<i64>)> = {
            let mut stmt = tx.prepare(
                "SELECT column_name, column_type, column_id, parent_column
                 FROM ducklake_column
                 WHERE table_id = ? AND end_snapshot IS NULL
                 ORDER BY column_order",
            )?;
            let mapped = stmt.query_map(params![table_id], |row| {
                let name: String = row.get(0)?;
                let col_type: String = row.get(1)?;
                let cid: i64 = row.get(2)?;
                let parent_column: Option<i64> = row.get(3)?;
                Ok((name, col_type, cid, parent_column))
            })?;
            mapped.collect::<std::result::Result<Vec<_>, duckdb::Error>>()?
        };

        let mut existing_catalog_columns = Vec::with_capacity(existing_rows.len());
        for (name, ducklake_type, column_id, parent_column) in existing_rows {
            existing_catalog_columns.push(ExistingCatalogColumn {
                column_id,
                name,
                ducklake_type,
                parent_column,
            });
        }
        let field_ids = assign_column_ids(&catalog_columns, &existing_catalog_columns, &fresh_ids)?;

        // Data-write policy: a data write — Replace OR Append — must NOT change a
        // column's type (that is schema evolution, which must go through
        // promote_column_type). Comparison is canonical (int64 ≡ bigint) so an
        // alias-only restatement is a no-op. Append additionally requires a
        // genuinely new column to be nullable. Mirrors the SQLite writer.
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

        // Final per-column ids: reuse the existing id for a column already in the
        // table, consume a freshly reserved id only for a genuinely new column.
        // These are baked into the staged parquet's field_id metadata, so they
        // must equal the ids finalize_snapshot commits. The column rows are
        // written at the commit point (not here).
        // Only the idempotent get-or-create schema/table rows are committed here;
        // they carry begin_snapshot = the reserved id and stay invisible until the
        // snapshot publishes, since schema/table reads ARE snapshot-scoped.
        tx.commit()?;

        Ok(WriteSetupResult {
            snapshot_id,
            base_snapshot_id,
            schema_id,
            table_id,
            column_ids: top_level_column_ids(&catalog_columns, &field_ids)?,
            field_ids,
        })
    }
}

fn enforce_inline_partition_fence(
    tx: &Transaction<'_>,
    table_id: i64,
    base_snapshot: i64,
) -> Result<()> {
    let changed: i64 = tx.query_row("SELECT COUNT(*) FROM ducklake_partition_info WHERE table_id = ? AND (begin_snapshot > ? OR end_snapshot > ?)", params![table_id, base_snapshot, base_snapshot], |row| row.get(0))?;
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
    use crate::DuckdbMetadataProvider;
    use crate::metadata_provider::MetadataProvider;
    use arrow::datatypes::{DataType, Field};
    use std::sync::Arc;
    use tempfile::TempDir;

    #[test]
    fn initialize_schema_migrates_directory_paths_once() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("catalog.ducklake");
        let writer = DuckdbMetadataWriter::new_with_init(db_path.to_str().unwrap()).unwrap();
        writer.set_data_path("/tmp/ducklake-data").unwrap();
        let snapshot_id = writer.create_snapshot().unwrap();
        let (schema_id, _) = writer
            .get_or_create_schema("main", None, snapshot_id)
            .unwrap();
        writer
            .get_or_create_table(schema_id, "users", None, snapshot_id)
            .unwrap();
        let paths = |writer: &DuckdbMetadataWriter| {
            let conn = writer.connection();
            let data_path: String = conn
                .query_row(
                    "SELECT value FROM ducklake_metadata WHERE key = 'data_path' AND scope IS NULL",
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            let schema_path: String = conn
                .query_row("SELECT path FROM ducklake_schema", [], |row| row.get(0))
                .unwrap();
            let table_path: String = conn
                .query_row("SELECT path FROM ducklake_table", [], |row| row.get(0))
                .unwrap();
            (data_path, schema_path, table_path)
        };
        let expected = (
            "/tmp/ducklake-data/".to_string(),
            "main/".to_string(),
            "users/".to_string(),
        );
        assert_eq!(paths(&writer), expected);

        writer
            .connection()
            .execute_batch(
                "UPDATE ducklake_metadata SET value = '/tmp/ducklake-data' WHERE key = 'data_path';
                 UPDATE ducklake_schema SET path = 'main';
                 UPDATE ducklake_table SET path = 'users';",
            )
            .unwrap();
        for _ in 0..2 {
            writer.initialize_schema().unwrap();
            assert_eq!(paths(&writer), expected);
        }
    }

    #[test]
    fn duckdb_writer_persists_recursive_columns() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("nested.ducklake");
        let writer = DuckdbMetadataWriter::new_with_init(db_path.to_str().unwrap()).unwrap();
        let levels = DataType::List(Arc::new(Field::new(
            "item",
            DataType::Struct(
                vec![
                    Arc::new(Field::new("price", DataType::Decimal128(38, 16), false)),
                    Arc::new(Field::new("count", DataType::UInt32, false)),
                ]
                .into(),
            ),
            false,
        )));
        let columns = vec![ColumnDef::from_arrow("bids", &levels, false).unwrap()];

        let setup = writer
            .begin_write_transaction("main", "depths", &columns, WriteMode::Replace)
            .unwrap();
        assert_eq!(setup.column_ids, vec![setup.field_ids[0]]);
        assert_eq!(setup.field_ids.len(), 4);
        writer
            .register_data_file(
                setup.table_id,
                "main",
                "depths",
                setup.snapshot_id,
                &DataFileInfo::new("depths.parquet", 1, 1),
                WriteMode::Replace,
                setup.base_snapshot_id,
                &columns,
                &setup.field_ids,
            )
            .unwrap();

        let mut connection = writer.connection();
        let transaction = connection.transaction().unwrap();
        let mut statement = transaction
            .prepare(
                "SELECT column_id, column_name, column_type, parent_column
                 FROM ducklake_column
                 WHERE table_id = ? AND end_snapshot IS NULL
                 ORDER BY column_order",
            )
            .unwrap();
        let rows = statement
            .query_map(params![setup.table_id], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<i64>>(3)?,
                ))
            })
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(
            rows,
            vec![
                (setup.field_ids[0], "bids".into(), "list".into(), None),
                (
                    setup.field_ids[1],
                    "element".into(),
                    "struct".into(),
                    Some(setup.field_ids[0]),
                ),
                (
                    setup.field_ids[2],
                    "price".into(),
                    "decimal(38, 16)".into(),
                    Some(setup.field_ids[1]),
                ),
                (
                    setup.field_ids[3],
                    "count".into(),
                    "uint32".into(),
                    Some(setup.field_ids[1]),
                ),
            ]
        );
    }

    fn test_writer(temp: &TempDir) -> DuckdbMetadataWriter {
        let writer = DuckdbMetadataWriter::new_with_init(
            temp.path().join("catalog.ducklake").to_str().unwrap(),
        )
        .unwrap();
        writer.set_data_path(temp.path().to_str().unwrap()).unwrap();
        writer
    }

    fn int_column() -> Vec<ColumnDef> {
        vec![ColumnDef::new("id", "int32", false).unwrap()]
    }

    fn append(
        writer: &DuckdbMetadataWriter,
        path: &str,
        rows: i64,
    ) -> (WriteSetupResult, CommitIds) {
        let columns = int_column();
        let setup = writer
            .begin_write_transaction("main", "t", &columns, WriteMode::Append)
            .unwrap();
        let commit = writer
            .register_data_file(
                setup.table_id,
                "main",
                "t",
                setup.snapshot_id,
                &DataFileInfo::new(path, rows * 10, rows),
                WriteMode::Append,
                setup.base_snapshot_id,
                &columns,
                &setup.column_ids,
            )
            .unwrap();
        (setup, commit)
    }

    fn file_id(writer: &DuckdbMetadataWriter, path: &str) -> i64 {
        writer
            .connection()
            .query_row(
                "SELECT data_file_id FROM ducklake_data_file WHERE path = ?",
                params![path],
                |row| row.get(0),
            )
            .unwrap()
    }

    fn append_inlined(
        writer: &DuckdbMetadataWriter,
        values: &[i32],
    ) -> (WriteSetupResult, CommitIds) {
        let columns = int_column();
        let setup = writer
            .begin_write_transaction("main", "t", &columns, WriteMode::Append)
            .unwrap();
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(values.to_vec()))])
            .unwrap();
        let commit = writer
            .register_inlined_data(
                setup.table_id,
                "main",
                "t",
                setup.snapshot_id,
                &[batch],
                WriteMode::Append,
                setup.base_snapshot_id,
                &columns,
                &setup.column_ids,
                &SnapshotCommitMetadata::default(),
                None,
            )
            .unwrap();
        (setup, commit)
    }

    fn inlined_table(writer: &DuckdbMetadataWriter, table_id: i64) -> String {
        writer
            .connection()
            .query_row(
                "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
                params![table_id],
                |row| row.get(0),
            )
            .unwrap()
    }

    fn changes_made(writer: &DuckdbMetadataWriter, snapshot_id: i64) -> Option<String> {
        writer
            .connection()
            .query_row(
                "SELECT changes_made FROM ducklake_snapshot_changes WHERE snapshot_id = ?",
                params![snapshot_id],
                |row| row.get(0),
            )
            .unwrap()
    }

    /// End-to-end round trip: write a small table through the real write path
    /// (begin_write_transaction + register_data_file), then read every catalog
    /// facet back through the DuckdbMetadataProvider and assert the snapshot,
    /// data_path, table, columns and data-file rows are byte-compatible.
    #[test]
    fn duckdb_write_then_read_back_via_provider() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("catalog.ducklake");
        let db_path_str = db_path.to_str().unwrap().to_string();
        let data_path = temp.path().join("data");
        let data_path_str = data_path.to_str().unwrap().to_string();

        // --- write ---
        {
            let writer = DuckdbMetadataWriter::new_with_init(&db_path_str).unwrap();
            writer.set_data_path(&data_path_str).unwrap();

            let columns = vec![
                ColumnDef::new("id", "int64", false).unwrap(),
                ColumnDef::new("name", "varchar", true).unwrap(),
            ];

            let setup = writer
                .begin_write_transaction("main", "users", &columns, WriteMode::Append)
                .unwrap();
            assert_eq!(setup.column_ids.len(), 2);
            assert_eq!(setup.base_snapshot_id, 0, "fresh catalog: no prior head");

            let file = DataFileInfo::new("users/data-0.parquet", 1024, 3).with_footer_size(256);
            let commit = writer
                .register_data_file(
                    setup.table_id,
                    "main",
                    "users",
                    setup.snapshot_id,
                    &file,
                    WriteMode::Append,
                    setup.base_snapshot_id,
                    &columns,
                    &setup.column_ids,
                )
                .unwrap();
            assert_eq!(commit.snapshot_id, 1, "first write commits snapshot 1");
            // writer (and its read-write lock on the file) dropped at block end.
        }

        // --- read back through the provider (read-only connection) ---
        let provider = DuckdbMetadataProvider::new(&db_path_str).unwrap();

        let snapshot = provider.get_current_snapshot().unwrap();
        assert_eq!(snapshot, 1, "committed head is snapshot 1");

        assert_eq!(
            provider.get_data_path().unwrap(),
            format!("{data_path_str}/")
        );

        let schema = provider
            .get_schema_by_name("main", snapshot)
            .unwrap()
            .expect("schema 'main' must exist");
        let table = provider
            .get_table_by_name(schema.schema_id, "users", snapshot)
            .unwrap()
            .expect("table 'users' must exist");
        assert_eq!(table.table_name, "users");

        let cols = provider
            .get_table_structure(table.table_id, snapshot)
            .unwrap();
        assert_eq!(cols.len(), 2, "both columns must read back");
        assert_eq!(cols[0].column_name, "id");
        assert_eq!(cols[0].column_type, "int64");
        assert!(!cols[0].is_nullable);
        assert_eq!(cols[1].column_name, "name");
        assert_eq!(cols[1].column_type, "varchar");
        assert!(cols[1].is_nullable);

        let files = provider
            .get_table_files_for_select(table.table_id, snapshot)
            .unwrap();
        assert_eq!(files.len(), 1, "exactly one data file");
        assert_eq!(files[0].file.path, "users/data-0.parquet");
        assert!(files[0].file.path_is_relative);
        assert_eq!(files[0].file.file_size_bytes, 1024);
        assert_eq!(files[0].file.footer_size, Some(256));
        assert_eq!(files[0].max_row_count, Some(3));
        assert_eq!(
            files[0].row_id_start,
            Some(0),
            "first file starts at rowid 0"
        );
    }

    /// A second Append hands out a fresh, non-overlapping rowid range and
    /// accumulates the table stats — mirrors the SQLite writer's
    /// `row_id_start_advances_across_inserts`.
    #[test]
    fn duckdb_row_id_start_advances_across_appends() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("catalog.ducklake");
        let db_path_str = db_path.to_str().unwrap().to_string();

        let writer = DuckdbMetadataWriter::new_with_init(&db_path_str).unwrap();
        writer.set_data_path("/tmp/does-not-matter").unwrap();
        let columns = vec![ColumnDef::new("id", "int64", false).unwrap()];

        let setup1 = writer
            .begin_write_transaction("main", "t", &columns, WriteMode::Append)
            .unwrap();
        writer
            .register_data_file(
                setup1.table_id,
                "main",
                "t",
                setup1.snapshot_id,
                &DataFileInfo::new("a.parquet", 100, 3),
                WriteMode::Append,
                setup1.base_snapshot_id,
                &columns,
                &setup1.column_ids,
            )
            .unwrap();

        let setup2 = writer
            .begin_write_transaction("main", "t", &columns, WriteMode::Append)
            .unwrap();
        let commit2 = writer
            .register_data_file(
                setup2.table_id,
                "main",
                "t",
                setup2.snapshot_id,
                &DataFileInfo::new("b.parquet", 250, 7),
                WriteMode::Append,
                setup2.base_snapshot_id,
                &columns,
                &setup2.column_ids,
            )
            .unwrap();
        assert_eq!(commit2.snapshot_id, 2, "second write commits snapshot 2");

        // Read the two files' row_id_start directly to assert non-overlapping ranges.
        let mut conn = writer.connection();
        let tx = conn.transaction().unwrap();
        let a_start: i64 = tx
            .query_row(
                "SELECT row_id_start FROM ducklake_data_file WHERE path = 'a.parquet'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let b_start: i64 = tx
            .query_row(
                "SELECT row_id_start FROM ducklake_data_file WHERE path = 'b.parquet'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let (records, next, bytes): (i64, i64, i64) = tx
            .query_row(
                "SELECT record_count, next_row_id, file_size_bytes
                 FROM ducklake_table_stats WHERE table_id = ?",
                params![setup1.table_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        tx.commit().unwrap();

        assert_eq!(a_start, 0, "first file starts at 0");
        assert_eq!(b_start, 3, "second file starts after the first file's rows");
        assert_eq!(records, 10, "record_count = 3 + 7");
        assert_eq!(next, 10, "next_row_id advances by sum of record_counts");
        assert_eq!(bytes, 350, "file_size_bytes accumulates");
    }

    #[test]
    fn duckdb_delete_update_and_truncate_share_sqlite_semantics() {
        let temp = TempDir::new().unwrap();
        let writer = test_writer(&temp);
        let (first_setup, first_commit) = append(&writer, "source.parquet", 4);
        let source_id = file_id(&writer, "source.parquet");
        let first_delete = writer
            .set_delete_file(
                first_setup.table_id,
                "main",
                "t",
                first_commit.snapshot_id + 1,
                source_id,
                None,
                first_commit.snapshot_id,
                &DeleteFileInfo::new("delete-1.parquet", 10, 1),
            )
            .unwrap();
        let first_delete_id: i64 = writer
            .connection()
            .query_row(
                "SELECT delete_file_id FROM ducklake_delete_file
                 WHERE data_file_id = ? AND end_snapshot IS NULL",
                params![source_id],
                |row| row.get(0),
            )
            .unwrap();

        let columns = int_column();
        let update = writer
            .begin_write_transaction("main", "t", &columns, WriteMode::Append)
            .unwrap();
        let update_commit = writer
            .register_data_file_with_deletes(
                update.table_id,
                "main",
                "t",
                update.snapshot_id,
                &DataFileInfo::new("updated.parquet", 10, 1),
                &[DeleteFileEntry {
                    data_file_id: source_id,
                    expected_prev_delete_file: Some(first_delete_id),
                    delete: DeleteFileInfo::new("delete-2.parquet", 20, 2),
                }],
                WriteMode::Append,
                update.base_snapshot_id,
                &columns,
                &update.column_ids,
            )
            .unwrap();
        assert!(writer.supports_update());
        assert_eq!(update_commit.snapshot_id, first_delete.snapshot_id + 1);
        let (new_file_snapshot, old_delete_end, new_delete_snapshot): (i64, Option<i64>, i64) =
            writer
                .connection()
                .query_row(
                    "SELECT
                    (SELECT begin_snapshot FROM ducklake_data_file WHERE path = 'updated.parquet'),
                    (SELECT end_snapshot FROM ducklake_delete_file WHERE delete_file_id = ?),
                    (SELECT begin_snapshot FROM ducklake_delete_file
                     WHERE path = 'delete-2.parquet')",
                    params![first_delete_id],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .unwrap();
        assert_eq!(new_file_snapshot, update_commit.snapshot_id);
        assert_eq!(old_delete_end, Some(update_commit.snapshot_id));
        assert_eq!(new_delete_snapshot, update_commit.snapshot_id);
        assert_eq!(
            changes_made(&writer, first_delete.snapshot_id),
            Some(format!("deleted_from_table:{}", first_setup.table_id))
        );
        assert_eq!(
            changes_made(&writer, update_commit.snapshot_id),
            Some(format!(
                "inserted_into_table:{0},deleted_from_table:{0}",
                first_setup.table_id
            ))
        );

        let removed = writer
            .commit_truncate(first_setup.table_id, "main", "t", update_commit.snapshot_id)
            .unwrap();
        assert_eq!(removed, 3);
        let state: (i64, i64, i64) = writer
            .connection()
            .query_row(
                "SELECT
                    (SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL),
                    (SELECT COUNT(*) FROM ducklake_delete_file WHERE end_snapshot IS NULL),
                    (SELECT record_count FROM ducklake_table_stats WHERE table_id = ?)",
                params![first_setup.table_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        // A truncate leaves the gross count of rows ever inserted.
        assert_eq!(state, (0, 0, 5));
    }

    #[test]
    fn duckdb_compaction_removes_sources_and_records_output() {
        let temp = TempDir::new().unwrap();
        let writer = test_writer(&temp);
        let (setup, first) = append(&writer, "a.parquet", 2);
        let (_, second) = append(&writer, "b.parquet", 3);
        let sources = [
            CompactionSourceFile {
                data_file_id: file_id(&writer, "a.parquet"),
                delete_file_id: None,
            },
            CompactionSourceFile {
                data_file_id: file_id(&writer, "b.parquet"),
                delete_file_id: None,
            },
        ];
        let output = CompactionOutputFile {
            file: DataFileInfo::new("merged.parquet", 45, 5),
            begin_snapshot: Some(first.snapshot_id),
            partial_max: Some(second.snapshot_id),
        };
        let commit = writer
            .commit_compaction(
                setup.table_id,
                second.snapshot_id,
                &sources,
                &[output],
                SourceRetirement::Remove,
            )
            .unwrap();
        let state: (i64, i64, Option<i64>, Option<i64>, i64, String) = writer
            .connection()
            .query_row(
                "SELECT
                    (SELECT COUNT(*) FROM ducklake_data_file),
                    (SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion),
                    (SELECT row_id_start FROM ducklake_data_file WHERE path = 'merged.parquet'),
                    (SELECT partial_max FROM ducklake_data_file WHERE path = 'merged.parquet'),
                    (SELECT record_count FROM ducklake_table_stats WHERE table_id = ?),
                    (SELECT changes_made FROM ducklake_snapshot_changes WHERE snapshot_id = ?)",
                params![setup.table_id, commit.snapshot_id],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(
            state,
            (
                1,
                2,
                None,
                Some(second.snapshot_id),
                5,
                format!("merge_adjacent:{}", setup.table_id),
            )
        );
    }

    #[test]
    fn duckdb_restore_retires_only_the_append_delta() {
        let temp = TempDir::new().unwrap();
        let writer = test_writer(&temp);
        let (setup, base) = append(&writer, "base.parquet", 2);
        append(&writer, "appended.parquet", 3);
        let restore = writer
            .retire_appends_since(setup.table_id, base.snapshot_id)
            .unwrap()
            .unwrap();
        let state: (Option<i64>, Option<i64>, i64, i64) = writer
            .connection()
            .query_row(
                "SELECT
                    (SELECT end_snapshot FROM ducklake_data_file WHERE path = 'base.parquet'),
                    (SELECT end_snapshot FROM ducklake_data_file WHERE path = 'appended.parquet'),
                    (SELECT record_count FROM ducklake_table_stats WHERE table_id = ?),
                    (SELECT next_row_id FROM ducklake_table_stats WHERE table_id = ?)",
                params![setup.table_id, setup.table_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        // Retiring leaves the gross count of rows ever inserted.
        assert_eq!(state, (None, Some(restore), 5, 5));
        assert_eq!(
            writer
                .retire_appends_since(setup.table_id, base.snapshot_id)
                .unwrap(),
            None
        );
    }

    #[test]
    fn duckdb_promotes_column_with_stable_identity() {
        let temp = TempDir::new().unwrap();
        let writer = test_writer(&temp);
        let (setup, _) = append(&writer, "a.parquet", 1);
        let old_column_id: i64 = writer
            .connection()
            .query_row(
                "SELECT column_id FROM ducklake_column WHERE end_snapshot IS NULL",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let snapshot = writer
            .promote_column_type(setup.table_id, "id", "int64")
            .unwrap();
        let versions: Vec<(i64, String, i64, Option<i64>)> = {
            let conn = writer.connection();
            let mut statement = conn
                .prepare(
                    "SELECT column_id, column_type, begin_snapshot, end_snapshot
                     FROM ducklake_column ORDER BY begin_snapshot",
                )
                .unwrap();
            statement
                .query_map([], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
                })
                .unwrap()
                .collect::<std::result::Result<Vec<_>, _>>()
                .unwrap()
        };
        assert_eq!(
            versions,
            vec![
                (old_column_id, "int32".to_string(), 1, Some(snapshot)),
                (old_column_id, "int64".to_string(), snapshot, None),
            ]
        );
        assert!(
            writer
                .promote_column_type(setup.table_id, "id", "int16")
                .is_err()
        );
    }

    #[test]
    fn duckdb_expiry_keeps_head_and_schedules_unreachable_files() {
        let temp = TempDir::new().unwrap();
        let writer = test_writer(&temp);
        let (setup, first) = append(&writer, "a.parquet", 2);
        writer
            .commit_truncate(setup.table_id, "main", "t", first.snapshot_id)
            .unwrap();
        let head: i64 = writer
            .connection()
            .query_row(
                "SELECT MAX(snapshot_id) FROM ducklake_snapshot",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let expired = writer
            .expire_snapshots(ExpireCriteria::Versions(vec![first.snapshot_id, head]))
            .unwrap();
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].snapshot_id, first.snapshot_id);
        let state: (i64, i64, i64) = writer
            .connection()
            .query_row(
                "SELECT
                    (SELECT COUNT(*) FROM ducklake_snapshot),
                    (SELECT COUNT(*) FROM ducklake_data_file),
                    (SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion)",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(state, (1, 0, 1));
    }

    /// `register_data_files_with_deletes` registers every appended file and
    /// every delete file in one snapshot, recording both kinds of change, and
    /// moves the table's row ids past every appended row.
    #[test]
    fn duckdb_register_data_files_with_deletes_commits_one_snapshot() {
        let temp = TempDir::new().unwrap();
        let writer = test_writer(&temp);
        let (setup, base) = append(&writer, "base.parquet", 4);
        let columns = int_column();
        let commit = writer
            .register_data_files_with_deletes(
                setup.table_id,
                "main",
                "t",
                0,
                &[
                    DataFileInfo::new("new_a.parquet", 20, 2),
                    DataFileInfo::new("new_b.parquet", 30, 3),
                ],
                &[DeleteFileEntry {
                    data_file_id: file_id(&writer, "base.parquet"),
                    expected_prev_delete_file: None,
                    delete: DeleteFileInfo::new("base_delete.parquet", 10, 2),
                }],
                WriteMode::Append,
                base.snapshot_id,
                &columns,
                &setup.column_ids,
            )
            .unwrap();
        assert_eq!(commit.snapshot_id, base.snapshot_id + 1);
        let (files, deletes, next_row_id): (i64, i64, i64) = writer
            .connection()
            .query_row(
                &format!(
                    "SELECT
                        (SELECT count(*) FROM ducklake_data_file
                         WHERE begin_snapshot = {0} AND path LIKE 'new_%'),
                        (SELECT count(*) FROM ducklake_delete_file
                         WHERE begin_snapshot = {0} AND path = 'base_delete.parquet'),
                        (SELECT next_row_id FROM ducklake_table_stats WHERE table_id = {1})",
                    commit.snapshot_id, setup.table_id
                ),
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!((files, deletes, next_row_id), (2, 1, 9));
        assert_eq!(
            changes_made(&writer, commit.snapshot_id),
            Some(format!(
                "inserted_into_table:{0},deleted_from_table:{0}",
                setup.table_id
            ))
        );
        let row_id_starts: Vec<i64> = writer
            .connection()
            .prepare(
                "SELECT row_id_start FROM ducklake_data_file WHERE path LIKE 'new_%' ORDER BY path",
            )
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<std::result::Result<_, _>>()
            .unwrap();
        assert_eq!(row_id_starts, vec![4, 6]);
    }

    /// Inlined DDL comes from the crate's own type mapping: catalog type text
    /// that does not parse as a DuckLake type is refused, never spliced.
    #[test]
    fn inlined_duckdb_type_is_built_from_the_type_mapping() {
        let decimal = DataType::Decimal128(18, 3);
        for hostile in [
            "decimal(18,3), pwned INTEGER",
            "decimal(18,3)); DROP TABLE ducklake_snapshot; --",
            "decimal(18,3) DEFAULT 1",
        ] {
            let error = inlined_duckdb_type(&decimal, hostile).unwrap_err();
            assert!(
                matches!(error, crate::DuckLakeError::UnsupportedType(_)),
                "{hostile}: {error}"
            );
        }
        assert_eq!(
            inlined_duckdb_type(&decimal, "decimal(18,3)").unwrap(),
            "DECIMAL(18, 3)"
        );
        assert_eq!(
            inlined_duckdb_type(&DataType::Int8, "int8").unwrap(),
            "TINYINT"
        );
        assert_eq!(
            inlined_duckdb_type(&DataType::FixedSizeBinary(16), "uuid").unwrap(),
            "UUID"
        );
        assert_eq!(
            inlined_duckdb_type(&DataType::Utf8, "json").unwrap(),
            "JSON"
        );
        assert_eq!(
            inlined_duckdb_value_sql(&DataType::UInt64, "uint64").unwrap(),
            "CAST(? AS UBIGINT)"
        );
    }

    #[test]
    fn duckdb_truncate_ends_inline_only_table() {
        let temp = TempDir::new().unwrap();
        let writer = test_writer(&temp);
        let (setup, commit) = append_inlined(&writer, &[1, 2]);
        let removed = writer
            .commit_truncate(setup.table_id, "main", "t", commit.snapshot_id)
            .unwrap();
        assert_eq!(removed, 2, "the two inline rows are the whole live table");
        let truncate_snapshot = commit.snapshot_id + 1;
        let physical = inlined_table(&writer, setup.table_id);
        let (head, live, ended, record_count): (i64, i64, i64, i64) = writer
            .connection()
            .query_row(
                &format!(
                    "SELECT
                        (SELECT MAX(snapshot_id) FROM ducklake_snapshot),
                        (SELECT COUNT(*) FROM {physical} WHERE end_snapshot IS NULL),
                        (SELECT COUNT(*) FROM {physical} WHERE end_snapshot = {truncate_snapshot}),
                        (SELECT record_count FROM ducklake_table_stats WHERE table_id = {})",
                    setup.table_id
                ),
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(
            (head, live, ended, record_count),
            (truncate_snapshot, 0, 2, 2)
        );
        assert_eq!(
            changes_made(&writer, truncate_snapshot),
            Some(format!("inlined_delete:{}", setup.table_id))
        );
        // Nothing left to truncate: idempotent no-op with no new snapshot.
        assert_eq!(
            writer
                .commit_truncate(setup.table_id, "main", "t", truncate_snapshot)
                .unwrap(),
            0
        );
    }

    #[test]
    fn duckdb_truncate_retires_files_and_ends_inline_rows() {
        let temp = TempDir::new().unwrap();
        let writer = test_writer(&temp);
        let (setup, _) = append(&writer, "base.parquet", 3);
        let (_, inline_commit) = append_inlined(&writer, &[10, 11]);
        let removed = writer
            .commit_truncate(setup.table_id, "main", "t", inline_commit.snapshot_id)
            .unwrap();
        assert_eq!(removed, 5, "3 parquet rows + 2 inline rows");
        let truncate_snapshot = inline_commit.snapshot_id + 1;
        let physical = inlined_table(&writer, setup.table_id);
        let (file_end, live_inline, ended_inline, record_count): (Option<i64>, i64, i64, i64) =
            writer
                .connection()
                .query_row(
                    &format!(
                        "SELECT
                            (SELECT end_snapshot FROM ducklake_data_file
                             WHERE path = 'base.parquet'),
                            (SELECT COUNT(*) FROM {physical} WHERE end_snapshot IS NULL),
                            (SELECT COUNT(*) FROM {physical}
                             WHERE end_snapshot = {truncate_snapshot}),
                            (SELECT record_count FROM ducklake_table_stats WHERE table_id = {})",
                        setup.table_id
                    ),
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
                )
                .unwrap();
        assert_eq!(
            (file_end, live_inline, ended_inline, record_count),
            (Some(truncate_snapshot), 0, 2, 5)
        );
        assert_eq!(
            changes_made(&writer, truncate_snapshot),
            Some(format!(
                "deleted_from_table:{0},inlined_delete:{0}",
                setup.table_id
            ))
        );
    }

    #[test]
    fn duckdb_commit_deletes_mixes_positional_and_inlined_in_one_snapshot() {
        let temp = TempDir::new().unwrap();
        let writer = test_writer(&temp);
        let (setup, _) = append(&writer, "base.parquet", 3);
        let (_, inline_commit) = append_inlined(&writer, &[10, 11]);
        let source_id = file_id(&writer, "base.parquet");
        let physical = inlined_table(&writer, setup.table_id);
        // Inline rows follow the parquet file's 3 rows: row_ids 3 and 4.
        let commit = writer
            .commit_deletes(
                setup.table_id,
                "main",
                "t",
                inline_commit.snapshot_id,
                &[DeleteFileEntry {
                    data_file_id: source_id,
                    expected_prev_delete_file: None,
                    delete: DeleteFileInfo::new("delete-1.parquet", 10, 1),
                }],
                &[InlinedRowRef {
                    table_name: physical.clone(),
                    row_id: 3,
                }],
            )
            .unwrap();
        assert_eq!(commit.snapshot_id, inline_commit.snapshot_id + 1);
        let (delete_begin, deleted_inline_end, other_inline_end, record_count): (
            i64,
            Option<i64>,
            Option<i64>,
            i64,
        ) = writer
            .connection()
            .query_row(
                &format!(
                    "SELECT
                        (SELECT begin_snapshot FROM ducklake_delete_file
                         WHERE path = 'delete-1.parquet'),
                        (SELECT end_snapshot FROM {physical} WHERE row_id = 3),
                        (SELECT end_snapshot FROM {physical} WHERE row_id = 4),
                        (SELECT record_count FROM ducklake_table_stats WHERE table_id = {})",
                    setup.table_id
                ),
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(delete_begin, commit.snapshot_id);
        assert_eq!(deleted_inline_end, Some(commit.snapshot_id));
        assert_eq!(other_inline_end, None);
        assert_eq!(record_count, 5, "deletes leave the gross count");
        assert_eq!(
            changes_made(&writer, commit.snapshot_id),
            Some(format!(
                "deleted_from_table:{0},inlined_delete:{0}",
                setup.table_id
            ))
        );
    }

    #[test]
    fn duckdb_positional_delete_snapshot_records_changes_made() {
        let temp = TempDir::new().unwrap();
        let writer = test_writer(&temp);
        let (setup, base) = append(&writer, "a.parquet", 2);
        let commit = writer
            .commit_positional_deletes(
                setup.table_id,
                "main",
                "t",
                base.snapshot_id,
                &[DeleteFileEntry {
                    data_file_id: file_id(&writer, "a.parquet"),
                    expected_prev_delete_file: None,
                    delete: DeleteFileInfo::new("d.parquet", 10, 1),
                }],
            )
            .unwrap();
        assert_eq!(
            changes_made(&writer, commit.snapshot_id),
            Some(format!("deleted_from_table:{}", setup.table_id))
        );
    }

    #[test]
    fn duckdb_first_write_commits_after_abandoned_staging() {
        let temp = TempDir::new().unwrap();
        let writer = test_writer(&temp);
        let columns = int_column();
        // Stage the table but never write: it persists with zero live columns.
        let staged = writer
            .begin_write_transaction("main", "t", &columns, WriteMode::Append)
            .unwrap();
        // The next write's commit creates the columns rather than conflicting.
        let (setup, commit) = append(&writer, "a.parquet", 2);
        assert_eq!(setup.table_id, staged.table_id);
        assert_eq!(commit.snapshot_id, 1, "abandoned staging left no snapshot");
        let live_columns: i64 = writer
            .connection()
            .query_row(
                "SELECT COUNT(*) FROM ducklake_column
                 WHERE table_id = ? AND end_snapshot IS NULL",
                params![setup.table_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(live_columns, 1);
    }
}
