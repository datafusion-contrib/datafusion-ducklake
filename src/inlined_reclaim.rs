//! Reclaiming catalog-inlined physical storage.
//!
//! Inlined rows live in metadata-database tables `ducklake_inlined_data_<table_id>_<version>`,
//! registered in `ducklake_inlined_data_tables`; inlined deletions of Parquet rows live in the
//! unregistered `ducklake_inlined_delete_<table_id>`. Official DuckLake
//! (`DuckLakeMetadataManager` / `DuckLakeTransactionState`) reclaims them at two points, which
//! the backends here port statement for statement:
//!
//! 1. **Expire, fully expired table** (`DeleteSnapshots`): every registered inlined data table of
//!    the table and its inlined delete table are dropped, and its registry rows are deleted with
//!    the rest of its per-table metadata.
//! 2. **After a flush commit** (`DropEmptySupersededInlinedTables`): a registered table whose
//!    schema version is below its table's newest registered version, and which holds no rows, is
//!    unregistered and dropped.
//!
//! Each backend runs these inside the transaction of the operation they belong to where its DDL
//! is transactional, so an interrupted run leaves nothing to finish. MySQL commits DDL implicitly;
//! its writer orders the statements so a later run finishes an interrupted one (see
//! `metadata_writer_mysql`).
//!
//! Like official, a pass drops whatever name a registry row holds, quoted as an identifier in
//! the backend's dialect. A catalog created before data inlining has no registry; every pass
//! checks for it first, with the capability probe the inlined readers use, and reclaims nothing
//! without it.

use crate::Result;
use crate::metadata_provider::inlined_delete_table_name;

/// Official's superseded-table query (`DropEmptySupersededInlinedTables`): every registered
/// inlined data table below its table's newest registered schema version.
pub(crate) const SUPERSEDED_INLINED_TABLES_SQL: &str =
    "SELECT idt.table_id, idt.schema_version, idt.table_name
     FROM ducklake_inlined_data_tables idt
     WHERE idt.schema_version < (
         SELECT MAX(idt2.schema_version)
         FROM ducklake_inlined_data_tables idt2
         WHERE idt2.table_id = idt.table_id)";

/// The physical tables official drops for fully expired tables: each registered inlined data
/// table, whatever its name, then one inlined delete table per table id. Callers quote each
/// name as an identifier.
pub(crate) fn dead_table_inlined_storage(
    registered: impl IntoIterator<Item = String>,
    dead_table_ids: &[i64],
) -> Result<Vec<String>> {
    let mut tables = registered.into_iter().collect::<Vec<_>>();
    for table_id in dead_table_ids {
        tables.push(inlined_delete_table_name(*table_id)?);
    }
    Ok(tables)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dead_table_storage_lists_registered_then_delete_tables() {
        let tables = dead_table_inlined_storage(
            [
                "ducklake_inlined_data_3_1".to_string(),
                "ducklake_snapshot; --".to_string(),
                "ducklake_inlined_data_3_2".to_string(),
            ],
            &[3, 4],
        )
        .unwrap();
        assert_eq!(
            tables,
            [
                "ducklake_inlined_data_3_1",
                "ducklake_snapshot; --",
                "ducklake_inlined_data_3_2",
                "ducklake_inlined_delete_3",
                "ducklake_inlined_delete_4",
            ]
        );
    }
}
