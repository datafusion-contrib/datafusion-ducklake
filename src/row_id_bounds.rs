//! Per-file `rowid` bounds on a multicatalog PostgreSQL store.
//!
//! **A divergence from official DuckLake, off by default.** A lookup by
//! `rowid` must find the files that may hold each id. The catalog says so for
//! a file whose `rowid`s are `row_id_start + position`, but not for one that
//! embeds them (an `UPDATE`'s row versions, an inlined flush, every
//! compaction output): only that file's footer does, so a lookup reads every
//! such footer. Official DuckLake records no bounds either; duckdb/ducklake
//! discussions #1389 and #1796 propose `row_id_min` / `row_id_max` columns on
//! `ducklake_data_file` for it.
//!
//! Until that lands, a store can keep the same bounds in a side table,
//! [`ROW_ID_BOUNDS_TABLE`], keyed by `data_file_id`, and no spec table
//! changes:
//!
//! - [`MulticatalogManager::create_row_id_bounds_table`] creates it. Nothing
//!   else does: a store without it is exactly a store before this module.
//! - A [`PostgresMetadataWriter`] built with `with_row_id_bounds(true)`
//!   records a file's bounds in the commit that registers it, from what the
//!   writer read of the footer it wrote ([`RowIdBounds`]). A file whose bounds
//!   are not known gets no row.
//! - A fork carries the source files' bounds to the rows it registers for
//!   them, when both its source provider and destination writer keep them.
//! - A [`MulticatalogProvider`] built with `with_row_id_bounds(true)` reads
//!   them, and [`DuckLakeTable::locate_rowids`] then skips, without opening
//!   it, every file whose bounds hold no requested id. A file without a row
//!   is read as before.
//!
//! The bounds need not be tight, only true: a file is immutable and a delete
//! only removes rows, so bounds recorded at registration hold for every
//! snapshot the file is visible in. Files written by anything that does not
//! record them (official DuckDB, a writer with them off, a release before
//! this one) simply have no row. A row of a file since removed names an id
//! no live file has, so it is never read; expire and compaction delete them
//! with their files all the same.
//!
//! When official DuckLake records the bounds, a store moves them into
//! `ducklake_data_file` and drops this table.
//!
//! [`MulticatalogManager::create_row_id_bounds_table`]: crate::MulticatalogManager::create_row_id_bounds_table
//! [`PostgresMetadataWriter`]: crate::PostgresMetadataWriter
//! [`MulticatalogProvider`]: crate::MulticatalogProvider
//! [`RowIdBounds`]: crate::metadata_writer::RowIdBounds
//! [`DuckLakeTable::locate_rowids`]: crate::table::DuckLakeTable::locate_rowids

/// The side table holding each data file's `rowid` bounds.
pub const ROW_ID_BOUNDS_TABLE: &str = "ducklake_catalog_data_file_row_id_bounds";

/// Creates [`ROW_ID_BOUNDS_TABLE`] when it does not exist.
pub(crate) const CREATE_ROW_ID_BOUNDS_TABLE: &str =
    "CREATE TABLE IF NOT EXISTS ducklake_catalog_data_file_row_id_bounds (
        data_file_id BIGINT PRIMARY KEY,
        row_id_min BIGINT NOT NULL,
        row_id_max BIGINT NOT NULL
    )";

/// Record `file`'s bounds for `data_file_id` in the commit `tx` when
/// `enabled` and they are known. `row_id_start` is the value the commit
/// stored for the file.
#[cfg(feature = "write-postgres")]
pub(crate) async fn insert_file_row_id_bounds(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    enabled: bool,
    data_file_id: i64,
    file: &crate::metadata_writer::DataFileInfo,
    row_id_start: Option<i64>,
) -> crate::Result<()> {
    if !enabled {
        return Ok(());
    }
    let Some((min, max)) = file.row_id_bounds.resolve(row_id_start, file.record_count) else {
        return Ok(());
    };
    sqlx::query(
        "INSERT INTO ducklake_catalog_data_file_row_id_bounds
             (data_file_id, row_id_min, row_id_max)
         VALUES ($1, $2, $3)",
    )
    .bind(data_file_id)
    .bind(min)
    .bind(max)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Delete the bounds of `data_file_ids` in `tx`, when `enabled`.
#[cfg(feature = "write-postgres")]
pub(crate) async fn delete_file_row_id_bounds(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    enabled: bool,
    data_file_ids: &[i64],
) -> crate::Result<()> {
    if !enabled || data_file_ids.is_empty() {
        return Ok(());
    }
    sqlx::query(
        "DELETE FROM ducklake_catalog_data_file_row_id_bounds WHERE data_file_id = ANY($1)",
    )
    .bind(data_file_ids)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Whether a file with `bounds` may hold an id of `sorted` (ascending). A
/// file without bounds may.
pub(crate) fn may_hold_any(bounds: Option<(i64, i64)>, sorted: &[i64]) -> bool {
    match bounds {
        Some((min, max)) => {
            let first = sorted.partition_point(|id| *id < min);
            sorted.get(first).is_some_and(|id| *id <= max)
        },
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata_writer::RowIdBounds;

    #[test]
    fn positional_bounds_follow_the_stored_start() {
        assert_eq!(RowIdBounds::Positional.resolve(Some(10), 5), Some((10, 14)));
        assert_eq!(RowIdBounds::Positional.resolve(None, 5), None);
        assert_eq!(RowIdBounds::Positional.resolve(Some(10), 0), None);
    }

    #[test]
    fn embedded_bounds_ignore_the_stored_start() {
        let bounds = RowIdBounds::Known {
            min: 0,
            max: 1,
        };
        assert_eq!(bounds.resolve(Some(6), 2), Some((0, 1)));
        assert_eq!(bounds.resolve(None, 2), Some((0, 1)));
    }

    #[test]
    fn unknown_bounds_record_nothing() {
        assert_eq!(RowIdBounds::Unknown.resolve(Some(0), 10), None);
    }

    #[test]
    fn a_file_without_bounds_may_hold_anything() {
        assert!(may_hold_any(None, &[1]));
        assert!(may_hold_any(None, &[]));
    }

    #[test]
    fn bounds_admit_only_ids_inside_them() {
        let sorted = [3, 20, 40];
        assert!(may_hold_any(Some((0, 3)), &sorted));
        assert!(may_hold_any(Some((15, 25)), &sorted));
        assert!(may_hold_any(Some((40, 40)), &sorted));
        assert!(!may_hold_any(Some((4, 19)), &sorted));
        assert!(!may_hold_any(Some((41, 100)), &sorted));
        assert!(!may_hold_any(Some((0, 2)), &sorted));
        assert!(!may_hold_any(Some((0, 100)), &[]));
    }
}
