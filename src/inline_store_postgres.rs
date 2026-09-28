//! Shared storage of catalog-inlined rows and deletions for the multicatalog
//! PostgreSQL layout.
//!
//! The DuckLake specification stores inlined rows in one physical table per
//! table and schema version (`ducklake_inlined_data_<table_id>_<schema_version>`)
//! and inlined deletions of Parquet rows in one table per table
//! (`ducklake_inlined_delete_<table_id>`). A multi-tenant store with many
//! catalogs, tables and schema versions then holds an unbounded number of
//! PostgreSQL relations (each with a TOAST table, an index, files, catalog rows
//! and autovacuum work), and the first inline write of a table or schema
//! version runs DDL under locks.
//!
//! The multicatalog layout is library-specific (DuckDB cannot read it), so it
//! keeps all inlined rows in `ducklake_inlined_row` and all inlined deletions in
//! `ducklake_inlined_file_delete`, keyed by `table_id`. The write path runs no
//! DDL. The single-catalog writers keep the specification layout.
//!
//! A row's values are one `BYTEA` cell, keyed by top-level `column_id` so that
//! renames, dropped and re-added columns and rows of older schema versions read
//! correctly:
//!
//! ```text
//! data   := version:u8 (=1) count:u32le cell*
//! cell   := column_id:i64le tag:u8 [len:u32le bytes]
//! tag    := 0 NULL | 1 TEXT (DuckLake literal text) | 2 RAW (string/binary bytes)
//! ```
//!
//! Strings and binary values are stored as their bytes; every other value as
//! the text the DuckLake literal parser reads back to the same value (floats in
//! shortest round-trip form, intervals as their three components). A column
//! that has no cell reads as its `initial_default`, the same as a Parquet file
//! written before the column existed.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::{DataType, IntervalUnit, SchemaRef};
use datafusion::common::ScalarValue;
use sqlx::postgres::{PgPool, Postgres};
use sqlx::{AssertSqlSafe, Row};

use crate::Result;
use crate::metadata_provider::{
    DuckLakeInlinedData, DuckLakeInlinedDelete, DuckLakeTableColumn, INLINED_DATA_REMEDIATION,
    build_inlined_batch, inlined_missing_scalar,
};

#[cfg(feature = "write-postgres")]
/// The shared tables and their indexes. `IF NOT EXISTS` throughout; run under
/// [`INIT_LOCK_KEY`] so concurrent initializations do not race on the catalog.
const SQL_CREATE_SHARED_INLINE_TABLES: &[&str] = &[
    // One row per version of an inlined row. An inline UPDATE keeps the row id
    // of the version it replaces, so a row id recurs with a later
    // begin_snapshot: the key includes it.
    r#"CREATE TABLE IF NOT EXISTS ducklake_inlined_row (
        table_id BIGINT NOT NULL,
        row_id BIGINT NOT NULL,
        begin_snapshot BIGINT NOT NULL,
        end_snapshot BIGINT,
        schema_version BIGINT NOT NULL,
        data BYTEA NOT NULL,
        PRIMARY KEY (table_id, row_id, begin_snapshot)
    )"#,
    // The live rows of one table: counts, truncate, and conflict checks.
    // Reads at a snapshot use the primary key's `table_id` prefix.
    r#"CREATE INDEX IF NOT EXISTS idx_inlined_row_live
        ON ducklake_inlined_row (table_id, begin_snapshot)
        WHERE end_snapshot IS NULL"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_inlined_file_delete (
        table_id BIGINT NOT NULL,
        file_id BIGINT NOT NULL,
        row_id BIGINT NOT NULL,
        begin_snapshot BIGINT NOT NULL,
        PRIMARY KEY (table_id, file_id, row_id)
    )"#,
];

#[cfg(feature = "write-postgres")]
/// Advisory transaction lock serializing creation of the shared tables and the
/// migration of the per-table layout.
const INIT_LOCK_KEY: &str = "datafusion-ducklake:multicatalog-inline-store";

const FORMAT_VERSION: u8 = 1;
const TAG_NULL: u8 = 0;
const TAG_TEXT: u8 = 1;
const TAG_RAW: u8 = 2;
/// Prefix of an interval cell: `months days nanoseconds`.
const INTERVAL_PREFIX: &str = "mdn:";

/// Logical name of the rows of `table_id` written at `schema_version`.
///
/// [`DuckLakeInlinedData::table_name`] and [`crate::metadata_writer::InlinedRowRef`]
/// carry it; it names no relation in this layout.
pub(crate) fn logical_table_name(table_id: i64, schema_version: i64) -> String {
    format!("ducklake_inlined_data_{table_id}_{schema_version}")
}

/// The schema version a [`logical_table_name`] of `table_id` names, or `None`
/// when the name belongs to another table or has another form.
#[cfg(feature = "write-postgres")]
pub(crate) fn parse_logical_table_name(table_id: i64, name: &str) -> Option<i64> {
    name.strip_prefix(&format!("ducklake_inlined_data_{table_id}_"))?
        .parse()
        .ok()
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

/// One decoded cell, borrowing from the row's bytes.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Cell<'a> {
    Null,
    Text(&'a str),
    Raw(&'a [u8]),
}

/// Encodes one row of `columns` (top-level arrays in catalog order) as a data
/// cell. `column_ids[i]` is the catalog id of `columns[i]`; `fill[i]` replaces
/// a NULL of that column with `fill_value` (the commit's snapshot id).
#[cfg(feature = "write-postgres")]
pub(crate) fn encode_row(
    columns: &[arrow::array::ArrayRef],
    column_ids: &[i64],
    row: usize,
    fill: &[bool],
    fill_value: i64,
) -> Result<Vec<u8>> {
    if columns.len() != column_ids.len() || columns.len() != fill.len() {
        return Err(crate::DuckLakeError::Internal(format!(
            "inlined row has {} columns for {} column ids",
            columns.len(),
            column_ids.len()
        )));
    }
    let count = u32::try_from(columns.len()).map_err(|_| {
        crate::DuckLakeError::Internal("inlined row has too many columns".to_string())
    })?;
    let mut out = Vec::with_capacity(5 + columns.len() * 16);
    out.push(FORMAT_VERSION);
    out.extend_from_slice(&count.to_le_bytes());
    for ((array, column_id), fill) in columns.iter().zip(column_ids).zip(fill) {
        out.extend_from_slice(&column_id.to_le_bytes());
        if *fill && array.is_null(row) {
            push_bytes(&mut out, TAG_TEXT, fill_value.to_string().as_bytes())?;
        } else {
            encode_cell(&mut out, array.as_ref(), row)?;
        }
    }
    Ok(out)
}

#[cfg(feature = "write-postgres")]
fn push_bytes(out: &mut Vec<u8>, tag: u8, bytes: &[u8]) -> Result<()> {
    let len = u32::try_from(bytes.len()).map_err(|_| {
        crate::DuckLakeError::Unsupported("inlined value exceeds 4 GiB".to_string())
    })?;
    out.push(tag);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}

#[cfg(feature = "write-postgres")]
fn encode_cell(out: &mut Vec<u8>, array: &dyn arrow::array::Array, row: usize) -> Result<()> {
    use arrow::array::AsArray;
    use arrow::datatypes::{Float32Type, Float64Type, IntervalMonthDayNanoType};

    if array.is_null(row) {
        out.push(TAG_NULL);
        return Ok(());
    }
    match array.data_type() {
        DataType::Utf8 => push_bytes(out, TAG_RAW, array.as_string::<i32>().value(row).as_bytes()),
        DataType::LargeUtf8 => {
            push_bytes(out, TAG_RAW, array.as_string::<i64>().value(row).as_bytes())
        },
        DataType::Utf8View => {
            push_bytes(out, TAG_RAW, array.as_string_view().value(row).as_bytes())
        },
        DataType::Binary => push_bytes(out, TAG_RAW, array.as_binary::<i32>().value(row)),
        DataType::LargeBinary => push_bytes(out, TAG_RAW, array.as_binary::<i64>().value(row)),
        DataType::BinaryView => push_bytes(out, TAG_RAW, array.as_binary_view().value(row)),
        DataType::FixedSizeBinary(_) => {
            push_bytes(out, TAG_RAW, array.as_fixed_size_binary().value(row))
        },
        // Rust formats a float in the shortest form that parses back to the same
        // bits, including `NaN`, `inf`, `-inf` and `-0`.
        DataType::Float32 => push_bytes(
            out,
            TAG_TEXT,
            array
                .as_primitive::<Float32Type>()
                .value(row)
                .to_string()
                .as_bytes(),
        ),
        DataType::Float64 => push_bytes(
            out,
            TAG_TEXT,
            array
                .as_primitive::<Float64Type>()
                .value(row)
                .to_string()
                .as_bytes(),
        ),
        DataType::Interval(IntervalUnit::MonthDayNano) => {
            let value = array.as_primitive::<IntervalMonthDayNanoType>().value(row);
            push_bytes(
                out,
                TAG_TEXT,
                format!(
                    "{INTERVAL_PREFIX}{} {} {}",
                    value.months, value.days, value.nanoseconds
                )
                .as_bytes(),
            )
        },
        DataType::List(_)
        | DataType::LargeList(_)
        | DataType::FixedSizeList(_, _)
        | DataType::Struct(_)
        | DataType::Map(_, _) => push_bytes(
            out,
            TAG_TEXT,
            crate::nested_inline::render_text(array, row)?.as_bytes(),
        ),
        _ => push_bytes(
            out,
            TAG_TEXT,
            arrow::util::display::array_value_to_string(array, row)?.as_bytes(),
        ),
    }
}

fn corrupt(detail: &str) -> crate::DuckLakeError {
    crate::DuckLakeError::Internal(format!("corrupt inlined row data: {detail}"))
}

/// Splits a data cell into `(column_id, cell)` pairs.
fn decode_cells(data: &[u8]) -> Result<Vec<(i64, Cell<'_>)>> {
    fn take<'a>(data: &mut &'a [u8], len: usize) -> Result<&'a [u8]> {
        if data.len() < len {
            return Err(corrupt("truncated"));
        }
        let (head, tail) = data.split_at(len);
        *data = tail;
        Ok(head)
    }
    let mut rest = data;
    let version = take(&mut rest, 1)?[0];
    if version != FORMAT_VERSION {
        return Err(corrupt(&format!("unknown format version {version}")));
    }
    let count = u32::from_le_bytes(take(&mut rest, 4)?.try_into().expect("4 bytes")) as usize;
    let mut cells = Vec::with_capacity(count.min(rest.len() / 9));
    for _ in 0..count {
        let column_id = i64::from_le_bytes(take(&mut rest, 8)?.try_into().expect("8 bytes"));
        let tag = take(&mut rest, 1)?[0];
        let cell = match tag {
            TAG_NULL => Cell::Null,
            TAG_TEXT | TAG_RAW => {
                let len =
                    u32::from_le_bytes(take(&mut rest, 4)?.try_into().expect("4 bytes")) as usize;
                let bytes = take(&mut rest, len)?;
                if tag == TAG_RAW {
                    Cell::Raw(bytes)
                } else {
                    Cell::Text(
                        std::str::from_utf8(bytes).map_err(|_| corrupt("text is not UTF-8"))?,
                    )
                }
            },
            other => return Err(corrupt(&format!("unknown cell tag {other}"))),
        };
        cells.push((column_id, cell));
    }
    if !rest.is_empty() {
        return Err(corrupt("trailing bytes"));
    }
    Ok(cells)
}

fn undecodable(
    column: &DuckLakeTableColumn,
    value: &str,
    data_type: &DataType,
) -> crate::DuckLakeError {
    crate::DuckLakeError::Unsupported(format!(
        "inlined data for column '{}' cannot decode value '{}' as {}; \
         {INLINED_DATA_REMEDIATION}",
        column.column_name, value, data_type
    ))
}

fn cell_scalar(
    cell: Cell<'_>,
    column: &DuckLakeTableColumn,
    data_type: &DataType,
) -> Result<ScalarValue> {
    let text = match cell {
        Cell::Null => return Ok(ScalarValue::try_from(data_type)?),
        Cell::Raw(bytes) => {
            let utf8 = || {
                String::from_utf8(bytes.to_vec())
                    .map_err(|_| undecodable(column, &String::from_utf8_lossy(bytes), data_type))
            };
            return Ok(match data_type {
                DataType::Utf8 => ScalarValue::Utf8(Some(utf8()?)),
                DataType::LargeUtf8 => ScalarValue::LargeUtf8(Some(utf8()?)),
                DataType::Utf8View => ScalarValue::Utf8View(Some(utf8()?)),
                DataType::Binary => ScalarValue::Binary(Some(bytes.to_vec())),
                DataType::LargeBinary => ScalarValue::LargeBinary(Some(bytes.to_vec())),
                DataType::BinaryView => ScalarValue::BinaryView(Some(bytes.to_vec())),
                DataType::FixedSizeBinary(size) if bytes.len() == *size as usize => {
                    ScalarValue::FixedSizeBinary(*size, Some(bytes.to_vec()))
                },
                _ => {
                    let text = utf8()?;
                    return crate::types::parse_ducklake_scalar(&text, data_type)
                        .ok_or_else(|| undecodable(column, &text, data_type));
                },
            });
        },
        Cell::Text(text) => text,
    };
    let exact = match data_type {
        DataType::Float64 => text
            .parse::<f64>()
            .ok()
            .map(|v| ScalarValue::Float64(Some(v))),
        DataType::Float32 => text
            .parse::<f32>()
            .ok()
            .map(|v| ScalarValue::Float32(Some(v))),
        DataType::Interval(IntervalUnit::MonthDayNano) => {
            text.strip_prefix(INTERVAL_PREFIX).and_then(|parts| {
                let mut parts = parts.split(' ');
                let months = parts.next()?.parse().ok()?;
                let days = parts.next()?.parse().ok()?;
                let nanoseconds = parts.next()?.parse().ok()?;
                parts.next().is_none().then(|| {
                    ScalarValue::IntervalMonthDayNano(Some(
                        arrow::datatypes::IntervalMonthDayNano::new(months, days, nanoseconds),
                    ))
                })
            })
        },
        _ => None,
    };
    match exact {
        Some(value) => Ok(value),
        None => crate::types::parse_ducklake_scalar(text, data_type)
            .ok_or_else(|| undecodable(column, text, data_type)),
    }
}

/// Builds a batch of `schema` (the Arrow form of `columns`) from data cells.
fn decode_batch(
    schema: SchemaRef,
    columns: &[DuckLakeTableColumn],
    rows: &[Vec<u8>],
) -> Result<RecordBatch> {
    let mut scalars = Vec::with_capacity(rows.len());
    for data in rows {
        let cells: HashMap<i64, Cell<'_>> = decode_cells(data)?.into_iter().collect();
        let row = columns
            .iter()
            .zip(schema.fields())
            .map(|(column, field)| match cells.get(&column.column_id) {
                Some(cell) => cell_scalar(*cell, column, field.data_type()),
                None => inlined_missing_scalar(column, field.data_type()),
            })
            .collect::<Result<Vec<_>>>()?;
        scalars.push(row);
    }
    build_inlined_batch(schema, columns, &scalars)
}

// ---------------------------------------------------------------------------
// Reads
// ---------------------------------------------------------------------------

/// Whether the shared tables exist. A store that no process of this version
/// has initialized yet keeps the per-table layout.
pub(crate) async fn shared_store_exists(pool: &PgPool) -> Result<bool> {
    Ok(sqlx::query_scalar(
        "SELECT to_regclass('ducklake_inlined_row') IS NOT NULL
            AND to_regclass('ducklake_inlined_file_delete') IS NOT NULL",
    )
    .fetch_one(pool)
    .await?)
}

const SQL_VISIBLE_ROWS: &str = "SELECT row_id, begin_snapshot, schema_version, data
     FROM ducklake_inlined_row
     WHERE table_id = $1 AND begin_snapshot <= $2
       AND (end_snapshot IS NULL OR end_snapshot > $2)";

/// Rows of `table_id` visible at `snapshot_id`, in row id order, as one batch.
pub(crate) async fn scan(
    pool: &PgPool,
    table_id: i64,
    snapshot_id: i64,
    columns: &[DuckLakeTableColumn],
) -> Result<Vec<RecordBatch>> {
    let rows = sqlx::query(AssertSqlSafe(format!("{SQL_VISIBLE_ROWS} ORDER BY row_id")))
        .bind(table_id)
        .bind(snapshot_id)
        .fetch_all(pool)
        .await?;
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let data = rows
        .iter()
        .map(|row| row.try_get::<Vec<u8>, _>(3))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let schema: SchemaRef = Arc::new(crate::types::build_arrow_schema(columns)?);
    Ok(vec![decode_batch(schema, columns, &data)?])
}

/// Rows of `table_id` visible at `snapshot_id` with their row ids, one entry
/// per schema version (named by [`logical_table_name`]), each ordered by
/// `(begin_snapshot, row_id)`.
pub(crate) async fn scan_with_row_ids(
    pool: &PgPool,
    table_id: i64,
    snapshot_id: i64,
    columns: &[DuckLakeTableColumn],
) -> Result<Vec<DuckLakeInlinedData>> {
    let rows = sqlx::query(AssertSqlSafe(format!(
        "{SQL_VISIBLE_ROWS} ORDER BY schema_version, begin_snapshot, row_id"
    )))
    .bind(table_id)
    .bind(snapshot_id)
    .fetch_all(pool)
    .await?;
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let schema: SchemaRef = Arc::new(crate::types::build_strict_arrow_schema(columns)?);
    /// Row ids, begin snapshots and data cells of one schema version.
    type Group = (Vec<i64>, Vec<i64>, Vec<Vec<u8>>);
    let mut groups: BTreeMap<i64, Group> = BTreeMap::new();
    for row in rows {
        let group = groups.entry(row.try_get(2)?).or_default();
        group.0.push(row.try_get(0)?);
        group.1.push(row.try_get(1)?);
        group.2.push(row.try_get(3)?);
    }
    groups
        .into_iter()
        .map(|(schema_version, (row_ids, begin_snapshots, data))| {
            Ok(DuckLakeInlinedData {
                table_name: logical_table_name(table_id, schema_version),
                row_ids,
                begin_snapshots,
                batch: decode_batch(schema.clone(), columns, &data)?,
            })
        })
        .collect()
}

/// Inlined deletions of Parquet rows of `table_id` visible at `snapshot_id`.
pub(crate) async fn file_deletes(
    pool: &PgPool,
    table_id: i64,
    snapshot_id: i64,
) -> Result<Vec<DuckLakeInlinedDelete>> {
    sqlx::query(
        "SELECT file_id, row_id FROM ducklake_inlined_file_delete
         WHERE table_id = $1 AND begin_snapshot <= $2
         ORDER BY file_id, row_id",
    )
    .bind(table_id)
    .bind(snapshot_id)
    .fetch_all(pool)
    .await?
    .into_iter()
    .map(|row| {
        Ok(DuckLakeInlinedDelete {
            data_file_id: row.try_get(0)?,
            row_id: row.try_get(1)?,
        })
    })
    .collect()
}

// ---------------------------------------------------------------------------
// Initialization and migration of the per-table layout
// ---------------------------------------------------------------------------

#[cfg(feature = "write-postgres")]
fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

#[cfg(feature = "write-postgres")]
/// Creates the shared tables when missing and moves the rows of every
/// per-table inlined relation (`ducklake_inlined_data_*` registered in
/// `ducklake_inlined_data_tables`, and `ducklake_inlined_delete_<table_id>`)
/// into them, then drops those relations. One transaction under an advisory
/// lock, so concurrent openers serialize and a crash leaves either layout
/// intact. Idempotent: once migrated, it only checks that there is nothing to do.
pub(crate) async fn initialize(pool: &PgPool) -> Result<()> {
    if shared_store_exists(pool).await? && !legacy_relations_exist(pool).await? {
        return Ok(());
    }
    let mut tx = pool.begin().await?;
    sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))")
        .bind(INIT_LOCK_KEY)
        .execute(&mut *tx)
        .await?;
    for statement in SQL_CREATE_SHARED_INLINE_TABLES {
        sqlx::query(*statement).execute(&mut *tx).await?;
    }
    migrate_legacy_rows(&mut tx).await?;
    migrate_legacy_deletes(&mut tx).await?;
    tx.commit().await?;
    Ok(())
}

#[cfg(feature = "write-postgres")]
async fn legacy_relations_exist(pool: &PgPool) -> Result<bool> {
    Ok(sqlx::query_scalar(
        r"SELECT EXISTS (
             SELECT 1 FROM pg_class c
             WHERE c.relkind = 'r' AND pg_table_is_visible(c.oid)
               AND (c.relname LIKE 'ducklake\_inlined\_delete\_%'
                    OR (c.relname LIKE 'ducklake\_inlined\_data\_%'
                        AND c.relname <> 'ducklake_inlined_data_tables'))
         )",
    )
    .fetch_one(pool)
    .await?)
}

#[cfg(feature = "write-postgres")]
/// A top-level column version of a table, for mapping a per-table relation's
/// column names to column ids.
struct ColumnVersion {
    column_id: i64,
    name: String,
    begin_snapshot: i64,
    end_snapshot: Option<i64>,
}

#[cfg(feature = "write-postgres")]
/// The column id a legacy value of column `name` in a row that began at
/// `begin_snapshot` belongs to: the version of that name live at the row's
/// snapshot, else the latest version of that name.
fn legacy_column_id(versions: &[ColumnVersion], name: &str, begin_snapshot: i64) -> Option<i64> {
    let named = || versions.iter().filter(|version| version.name == name);
    named()
        .find(|version| {
            version.begin_snapshot <= begin_snapshot
                && version.end_snapshot.is_none_or(|end| begin_snapshot < end)
        })
        .or_else(|| named().max_by_key(|version| version.begin_snapshot))
        .map(|version| version.column_id)
}

#[cfg(feature = "write-postgres")]
async fn migrate_legacy_rows(tx: &mut sqlx::Transaction<'_, Postgres>) -> Result<()> {
    let registry_exists: bool =
        sqlx::query_scalar("SELECT to_regclass('ducklake_inlined_data_tables') IS NOT NULL")
            .fetch_one(&mut **tx)
            .await?;
    if !registry_exists {
        return Ok(());
    }
    let registered = sqlx::query(
        "SELECT table_id, table_name, schema_version FROM ducklake_inlined_data_tables",
    )
    .fetch_all(&mut **tx)
    .await?;
    for entry in registered {
        let table_id: i64 = entry.try_get(0)?;
        let name: String = entry.try_get(1)?;
        let schema_version: Option<i64> = entry.try_get(2)?;
        if crate::metadata_provider::is_inlined_data_table(&name) {
            let exists: bool = sqlx::query_scalar(
                "SELECT EXISTS (SELECT 1 FROM pg_class c
                 WHERE c.relkind = 'r' AND c.relname = $1 AND pg_table_is_visible(c.oid))",
            )
            .bind(&name)
            .fetch_one(&mut **tx)
            .await?;
            if exists {
                migrate_legacy_table(tx, table_id, &name, schema_version.unwrap_or(0)).await?;
                sqlx::query(AssertSqlSafe(format!("DROP TABLE {}", quote_ident(&name))))
                    .execute(&mut **tx)
                    .await?;
            }
        }
        sqlx::query(
            "DELETE FROM ducklake_inlined_data_tables WHERE table_id = $1 AND table_name = $2",
        )
        .bind(table_id)
        .bind(&name)
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

#[cfg(feature = "write-postgres")]
/// Moves the rows of one `ducklake_inlined_data_*` relation. A `BYTEA` value
/// (strings and binary) moves as its bytes, any other as PostgreSQL's text of
/// it, which is what the per-table reader parsed. Rows of a table the catalog
/// no longer has are dropped with the relation.
async fn migrate_legacy_table(
    tx: &mut sqlx::Transaction<'_, Postgres>,
    table_id: i64,
    name: &str,
    schema_version: i64,
) -> Result<()> {
    let versions = sqlx::query(
        "SELECT column_id, column_name, begin_snapshot, end_snapshot FROM ducklake_column
         WHERE table_id = $1 AND parent_column IS NULL",
    )
    .bind(table_id)
    .fetch_all(&mut **tx)
    .await?
    .into_iter()
    .map(|row| {
        Ok(ColumnVersion {
            column_id: row.try_get(0)?,
            name: row.try_get(1)?,
            begin_snapshot: row.try_get(2)?,
            end_snapshot: row.try_get(3)?,
        })
    })
    .collect::<Result<Vec<_>>>()?;
    if versions.is_empty() {
        return Ok(());
    }
    let physical = sqlx::query(
        "SELECT column_name::TEXT, data_type::TEXT FROM information_schema.columns
         WHERE table_schema = current_schema() AND table_name = $1
           AND column_name NOT IN ('row_id', 'begin_snapshot', 'end_snapshot')
         ORDER BY ordinal_position",
    )
    .bind(name)
    .fetch_all(&mut **tx)
    .await?
    .into_iter()
    .map(|row| {
        Ok((
            row.try_get::<String, _>(0)?,
            row.try_get::<String, _>(1)? == "bytea",
        ))
    })
    .collect::<Result<Vec<_>>>()?;
    let projection = physical
        .iter()
        .map(|(column, bytea)| {
            if *bytea {
                quote_ident(column)
            } else {
                format!("CAST({} AS TEXT)", quote_ident(column))
            }
        })
        .chain(std::iter::once("NULL::TEXT".to_string()))
        .collect::<Vec<_>>()
        .join(", ");
    let rows = sqlx::query(AssertSqlSafe(format!(
        "SELECT row_id, begin_snapshot, end_snapshot, {projection} FROM {}
         WHERE row_id IS NOT NULL AND begin_snapshot IS NOT NULL",
        quote_ident(name)
    )))
    .fetch_all(&mut **tx)
    .await?;

    let mut row_ids = Vec::with_capacity(rows.len());
    let mut begins = Vec::with_capacity(rows.len());
    let mut ends = Vec::with_capacity(rows.len());
    let mut data = Vec::with_capacity(rows.len());
    for row in rows {
        let begin_snapshot: i64 = row.try_get(1)?;
        let mut cells = Vec::new();
        let mut count = 0u32;
        for (index, (column, bytea)) in physical.iter().enumerate() {
            let Some(column_id) = legacy_column_id(&versions, column, begin_snapshot) else {
                continue;
            };
            cells.extend_from_slice(&column_id.to_le_bytes());
            if *bytea {
                match row.try_get::<Option<Vec<u8>>, _>(index + 3)? {
                    Some(bytes) => push_bytes(&mut cells, TAG_RAW, &bytes)?,
                    None => cells.push(TAG_NULL),
                }
            } else {
                match row.try_get::<Option<String>, _>(index + 3)? {
                    Some(text) => push_bytes(&mut cells, TAG_TEXT, text.as_bytes())?,
                    None => cells.push(TAG_NULL),
                }
            }
            count += 1;
        }
        let mut encoded = Vec::with_capacity(5 + cells.len());
        encoded.push(FORMAT_VERSION);
        encoded.extend_from_slice(&count.to_le_bytes());
        encoded.extend_from_slice(&cells);
        row_ids.push(row.try_get::<i64, _>(0)?);
        begins.push(begin_snapshot);
        ends.push(row.try_get::<Option<i64>, _>(2)?);
        data.push(encoded);
    }
    sqlx::query(
        "INSERT INTO ducklake_inlined_row
             (table_id, row_id, begin_snapshot, end_snapshot, schema_version, data)
         SELECT $1, r, b, e, $2, d
         FROM UNNEST($3::BIGINT[], $4::BIGINT[], $5::BIGINT[], $6::BYTEA[]) AS u(r, b, e, d)
         ON CONFLICT DO NOTHING",
    )
    .bind(table_id)
    .bind(schema_version)
    .bind(&row_ids)
    .bind(&begins)
    .bind(&ends)
    .bind(&data)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

#[cfg(feature = "write-postgres")]
async fn migrate_legacy_deletes(tx: &mut sqlx::Transaction<'_, Postgres>) -> Result<()> {
    let names: Vec<String> = sqlx::query_scalar(
        r"SELECT c.relname::TEXT FROM pg_class c
          WHERE c.relkind = 'r' AND pg_table_is_visible(c.oid)
            AND c.relname LIKE 'ducklake\_inlined\_delete\_%'",
    )
    .fetch_all(&mut **tx)
    .await?;
    for name in names {
        let Some(table_id) = name
            .strip_prefix("ducklake_inlined_delete_")
            .and_then(|id| id.parse::<i64>().ok())
        else {
            continue;
        };
        let quoted = quote_ident(&name);
        sqlx::query(AssertSqlSafe(format!(
            "INSERT INTO ducklake_inlined_file_delete (table_id, file_id, row_id, begin_snapshot)
             SELECT $1, file_id, row_id, MIN(begin_snapshot) FROM {quoted}
             WHERE file_id IS NOT NULL AND row_id IS NOT NULL AND begin_snapshot IS NOT NULL
             GROUP BY file_id, row_id
             ON CONFLICT DO NOTHING"
        )))
        .bind(table_id)
        .execute(&mut **tx)
        .await?;
        sqlx::query(AssertSqlSafe(format!("DROP TABLE {quoted}")))
            .execute(&mut **tx)
            .await?;
    }
    Ok(())
}

#[cfg(all(test, feature = "write-postgres"))]
mod tests {
    use super::*;
    use arrow::array::{
        ArrayRef, BinaryArray, BooleanArray, Date32Array, Decimal128Array, FixedSizeBinaryArray,
        Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
        IntervalMonthDayNanoArray, LargeStringArray, ListArray, StringArray,
        Time64MicrosecondArray, TimestampMicrosecondArray, TimestampMillisecondArray,
        TimestampNanosecondArray, TimestampSecondArray, UInt8Array, UInt16Array, UInt32Array,
        UInt64Array,
    };
    use arrow::datatypes::{Int32Type, IntervalMonthDayNano};

    fn column(id: i64, name: &str, ducklake_type: &str) -> DuckLakeTableColumn {
        DuckLakeTableColumn::new(id, name.to_string(), ducklake_type.to_string(), true)
    }

    /// Every array round-trips through encode and decode, bit for bit.
    #[test]
    fn values_round_trip_exactly() {
        let arrays: Vec<(&str, ArrayRef)> = vec![
            (
                "boolean",
                Arc::new(BooleanArray::from(vec![Some(true), Some(false), None])),
            ),
            (
                "int8",
                Arc::new(Int8Array::from(vec![Some(i8::MIN), Some(i8::MAX), None])),
            ),
            (
                "int16",
                Arc::new(Int16Array::from(vec![Some(i16::MIN), Some(i16::MAX), None])),
            ),
            (
                "int32",
                Arc::new(Int32Array::from(vec![Some(i32::MIN), Some(i32::MAX), None])),
            ),
            (
                "int64",
                Arc::new(Int64Array::from(vec![Some(i64::MIN), Some(i64::MAX), None])),
            ),
            (
                "uint8",
                Arc::new(UInt8Array::from(vec![Some(0), Some(u8::MAX), None])),
            ),
            (
                "uint16",
                Arc::new(UInt16Array::from(vec![Some(0), Some(u16::MAX), None])),
            ),
            (
                "uint32",
                Arc::new(UInt32Array::from(vec![Some(0), Some(u32::MAX), None])),
            ),
            (
                "uint64",
                Arc::new(UInt64Array::from(vec![Some(0), Some(u64::MAX), None])),
            ),
            (
                "float",
                Arc::new(Float32Array::from(vec![
                    Some(f32::NAN),
                    Some(f32::INFINITY),
                    Some(f32::NEG_INFINITY),
                    Some(-0.0),
                    Some(0.1),
                    Some(f32::MIN_POSITIVE / 3.0),
                    Some(f32::MAX),
                    None,
                ])),
            ),
            (
                "double",
                Arc::new(Float64Array::from(vec![
                    Some(f64::NAN),
                    Some(f64::INFINITY),
                    Some(f64::NEG_INFINITY),
                    Some(-0.0),
                    Some(0.1 + 0.2),
                    Some(f64::MIN_POSITIVE / 3.0),
                    Some(f64::MAX),
                    Some(1e-300),
                    None,
                ])),
            ),
            (
                "decimal(38,10)",
                Arc::new(
                    Decimal128Array::from(vec![
                        Some(i128::from(i64::MAX) * 1_000_000_007),
                        Some(-1),
                        Some(0),
                        None,
                    ])
                    .with_precision_and_scale(38, 10)
                    .unwrap(),
                ),
            ),
            (
                "date",
                Arc::new(Date32Array::from(vec![
                    Some(-719_162),
                    Some(0),
                    Some(2_932_896),
                    None,
                ])),
            ),
            (
                "time",
                Arc::new(Time64MicrosecondArray::from(vec![
                    Some(0),
                    Some(86_399_999_999),
                    None,
                ])),
            ),
            (
                "timestamp_s",
                Arc::new(TimestampSecondArray::from(vec![
                    Some(-62_135_596_800),
                    Some(1),
                    None,
                ])),
            ),
            (
                "timestamp_ms",
                Arc::new(TimestampMillisecondArray::from(vec![
                    Some(-1),
                    Some(1_700_000_000_123),
                    None,
                ])),
            ),
            (
                "timestamp",
                Arc::new(TimestampMicrosecondArray::from(vec![
                    Some(-1),
                    Some(1_700_000_000_123_456),
                    None,
                ])),
            ),
            (
                "timestamptz",
                Arc::new(
                    TimestampMicrosecondArray::from(vec![
                        Some(-1),
                        Some(1_700_000_000_123_456),
                        None,
                    ])
                    .with_timezone("UTC"),
                ),
            ),
            (
                "timestamp_ns",
                Arc::new(TimestampNanosecondArray::from(vec![
                    Some(i64::MIN + 1),
                    Some(1_700_000_000_123_456_789),
                    None,
                ])),
            ),
            (
                "timestamptz_ns",
                Arc::new(
                    TimestampNanosecondArray::from(vec![
                        Some(-1),
                        Some(1_700_000_000_123_456_789),
                        None,
                    ])
                    .with_timezone("UTC"),
                ),
            ),
            (
                "interval",
                Arc::new(IntervalMonthDayNanoArray::from(vec![
                    Some(IntervalMonthDayNano::new(-13, 40, -1)),
                    Some(IntervalMonthDayNano::new(1, 2, 3_000_000_007)),
                    None,
                ])),
            ),
            (
                "varchar",
                Arc::new(StringArray::from(vec![
                    Some("plain"),
                    Some("nul \0 byte, quote ' and \" and \\ and ünïcødé"),
                    Some(""),
                    None,
                ])),
            ),
            (
                "varchar",
                Arc::new(LargeStringArray::from(vec![Some("large"), None])),
            ),
            (
                "blob",
                Arc::new(BinaryArray::from(vec![
                    Some(&[0u8, 255, 0x5c, 0x78][..]),
                    Some(&[][..]),
                    None,
                ])),
            ),
            (
                "uuid",
                Arc::new(
                    FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                        vec![Some([0x55u8; 16]), None].into_iter(),
                        16,
                    )
                    .unwrap(),
                ),
            ),
            (
                "int32[]",
                Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
                    Some(vec![Some(1), None, Some(-3)]),
                    Some(vec![]),
                    None,
                ])),
            ),
        ];
        for (ducklake_type, array) in arrays {
            let columns = vec![column(7, "c", ducklake_type)];
            let data_type = array.data_type().clone();
            let schema: SchemaRef = Arc::new(arrow::datatypes::Schema::new(vec![
                arrow::datatypes::Field::new("c", data_type.clone(), true),
            ]));
            let rows = (0..array.len())
                .map(|row| {
                    encode_row(std::slice::from_ref(&array), &[7], row, &[false], 0).unwrap()
                })
                .collect::<Vec<_>>();
            let batch = decode_batch(schema, &columns, &rows)
                .unwrap_or_else(|error| panic!("{ducklake_type}: {error}"));
            let decoded = batch.column(0);
            assert_eq!(decoded.data_type(), &data_type, "{ducklake_type}");
            assert_eq!(decoded.len(), array.len(), "{ducklake_type}");
            for row in 0..array.len() {
                let expected = ScalarValue::try_from_array(array.as_ref(), row).unwrap();
                let actual = ScalarValue::try_from_array(decoded.as_ref(), row).unwrap();
                // Compare floats by bits: NaN != NaN and 0.0 == -0.0 under ==.
                let same = match (&expected, &actual) {
                    (ScalarValue::Float32(Some(a)), ScalarValue::Float32(Some(b))) => {
                        a.to_bits() == b.to_bits()
                    },
                    (ScalarValue::Float64(Some(a)), ScalarValue::Float64(Some(b))) => {
                        a.to_bits() == b.to_bits()
                    },
                    _ => expected == actual,
                };
                assert!(
                    same,
                    "{ducklake_type} row {row}: {expected:?} != {actual:?}"
                );
            }
        }
    }

    /// A column the row has no cell for reads as its initial default; a cell
    /// of a column the reader does not ask for is ignored.
    #[test]
    fn missing_and_extra_cells() {
        let ids: ArrayRef = Arc::new(Int32Array::from(vec![1]));
        let gone: ArrayRef = Arc::new(Int32Array::from(vec![2]));
        let data = encode_row(&[ids, gone], &[1, 2], 0, &[false, false], 0).unwrap();
        let mut added = column(3, "added", "int32");
        added.initial_default = Some("42".to_string());
        let columns = vec![column(1, "id", "int32"), added, column(4, "empty", "int32")];
        let schema: SchemaRef = Arc::new(crate::types::build_arrow_schema(&columns).unwrap());
        let batch = decode_batch(schema, &columns, &[data]).unwrap();
        let values = (0..3)
            .map(|index| ScalarValue::try_from_array(batch.column(index), 0).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            values,
            vec![
                ScalarValue::Int32(Some(1)),
                ScalarValue::Int32(Some(42)),
                ScalarValue::Int32(None)
            ]
        );
    }

    #[test]
    fn corrupt_data_is_an_error() {
        let ids: ArrayRef = Arc::new(Int32Array::from(vec![1]));
        let data = encode_row(&[ids], &[1], 0, &[false], 0).unwrap();
        for bad in [&data[..data.len() - 1], &[9u8, 0, 0, 0, 0][..], &[][..]] {
            assert!(decode_cells(bad).is_err());
        }
        let mut trailing = data.clone();
        trailing.push(0);
        assert!(decode_cells(&trailing).is_err());
    }

    /// Prints the cost of encoding and decoding a row of eight mixed columns.
    /// `cargo test --release --features write-postgres --lib encoding_cost -- --ignored --nocapture`
    #[test]
    #[ignore = "timing, not a correctness check"]
    fn encoding_cost() {
        const ROWS: usize = 10_000;
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from_iter_values(0..ROWS as i64)),
            Arc::new(Float64Array::from_iter_values(
                (0..ROWS).map(|v| v as f64 / 3.0),
            )),
            Arc::new(StringArray::from_iter_values(
                (0..ROWS).map(|v| format!("name {v}")),
            )),
            Arc::new(
                Decimal128Array::from_iter_values((0..ROWS).map(|v| v as i128 * 1_001))
                    .with_precision_and_scale(18, 3)
                    .unwrap(),
            ),
            Arc::new(Date32Array::from_iter_values((0..ROWS).map(|v| v as i32))),
            Arc::new(
                TimestampMicrosecondArray::from_iter_values(
                    (0..ROWS).map(|v| 1_700_000_000_000_000 + v as i64),
                )
                .with_timezone("UTC"),
            ),
            Arc::new(BooleanArray::from_iter((0..ROWS).map(|v| Some(v % 2 == 0)))),
            Arc::new(BinaryArray::from_iter_values(
                (0..ROWS).map(|v| v.to_le_bytes()),
            )),
        ];
        let types = [
            "int64",
            "double",
            "varchar",
            "decimal(18,3)",
            "date",
            "timestamptz",
            "boolean",
            "blob",
        ];
        let columns = types
            .iter()
            .enumerate()
            .map(|(index, ducklake_type)| column(index as i64, &format!("c{index}"), ducklake_type))
            .collect::<Vec<_>>();
        let ids = (0..types.len() as i64).collect::<Vec<_>>();
        let fill = vec![false; types.len()];
        let started = std::time::Instant::now();
        let rows = (0..ROWS)
            .map(|row| encode_row(&arrays, &ids, row, &fill, 0).unwrap())
            .collect::<Vec<_>>();
        let encode = started.elapsed();
        let schema: SchemaRef = Arc::new(crate::types::build_arrow_schema(&columns).unwrap());
        let started = std::time::Instant::now();
        let batch = decode_batch(schema, &columns, &rows).unwrap();
        let decode = started.elapsed();
        assert_eq!(batch.num_rows(), ROWS);
        let bytes: usize = rows.iter().map(Vec::len).sum();
        println!(
            "8 mixed columns: encode {:?}/row, decode {:?}/row, {} bytes/row",
            encode / ROWS as u32,
            decode / ROWS as u32,
            bytes / ROWS
        );
    }

    #[test]
    fn logical_names_round_trip() {
        assert_eq!(
            parse_logical_table_name(12, &logical_table_name(12, 3)),
            Some(3)
        );
        assert_eq!(
            parse_logical_table_name(1, &logical_table_name(12, 3)),
            None
        );
        assert_eq!(
            parse_logical_table_name(12, "ducklake_inlined_data_12_x"),
            None
        );
    }

    #[test]
    fn legacy_columns_map_by_the_version_live_at_the_row() {
        let versions = vec![
            ColumnVersion {
                column_id: 1,
                name: "a".to_string(),
                begin_snapshot: 1,
                end_snapshot: Some(5),
            },
            ColumnVersion {
                column_id: 2,
                name: "a".to_string(),
                begin_snapshot: 6,
                end_snapshot: None,
            },
        ];
        assert_eq!(legacy_column_id(&versions, "a", 3), Some(1));
        assert_eq!(legacy_column_id(&versions, "a", 7), Some(2));
        assert_eq!(legacy_column_id(&versions, "a", 5), Some(2));
        assert_eq!(legacy_column_id(&versions, "b", 3), None);
    }
}
