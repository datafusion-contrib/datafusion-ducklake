//! Single integration-test binary.
//!
//! Every file under `tests/it/` is a module here rather than its own `tests/*.rs`
//! target. Cargo links one test binary per top-level file, and with
//! `duckdb-bundled` on by default each of those statically linked its own copy of
//! DuckDB — ~50 binaries at 260-430 MB apiece, several gigabytes of linker output
//! per build. That is what exhausted the runner disk during linking (`lld` dying
//! with SIGBUS, hence `CARGO_PROFILE_TEST_DEBUG: "0"` in CI) and what made the
//! cached target directory too large to fit GitHub's 10 GB per-repository budget.
//!
//! Each module keeps its own `#![cfg(feature = ...)]` inner attribute, so the
//! feature gating is unchanged: when a feature is off the module compiles to
//! nothing, exactly as the standalone binary previously compiled to zero tests.
//!
//! Test names gain a module prefix (`write_tests::test_append_semantics`), which
//! substring filters like `cargo test delete_filter` still match.

mod append_with_deletes_postgres_tests;
mod append_with_deletes_tests;
mod cdc_cumulative_delete_tests;
mod cdc_differential_tests;
mod cdc_rowid_tests;
mod column_defaults_tests;
mod column_stats_tests;
mod common;
mod compaction_postgres_tests;
mod compaction_sqlite_tests;
mod concurrent_staged_upload_tests;
mod concurrent_tests;
mod concurrent_write_tests;
mod delete_filter_tests;
mod empty_data_file_tests;
mod encryption_tests;
mod files_matching_tests;
mod hybrid_asyncdb;
mod information_schema_test;
mod inlined_data_backends_tests;
mod inlined_data_sqlite_tests;
mod inlined_delete_fixture;
mod inlined_delete_tests;
mod inlined_native_types_tests;
mod insert_partitioning_tests;
mod keyed_mutation_after_compaction_tests;
mod maintenance_sqlite_tests;
mod metadata_contract_tests;
mod missing_delete_file_tests;
mod multicatalog_hardening_tests;
mod multicatalog_postgres_tests;
mod multicatalog_provider_tests;
mod mysql_metadata_provider_test;
mod mysql_metadata_writer_test;
mod name_mapping_tests;
mod nested_field_id_schema_tests;
mod numeric_metadata_validation_tests;
mod object_store_integration_test;
mod partition_tests;
mod partition_write_duckdb_tests;
mod partition_write_tests;
mod positional_delete_oracle_postgres_tests;
mod positional_delete_oracle_tests;
mod positional_delete_tests;
mod positional_pushdown_tests;
mod postgres_metadata_provider_test;
mod postgres_single_catalog_write_tests;
mod renamed_columns_tests;
mod row_count_tests;
mod row_id_tests;
mod rowid_physical_position_tests;
mod scoped_settings_tests;
mod sorted_write_duckdb_tests;
mod sorted_write_tests;
mod sql_delete_postgres_tests;
mod sql_delete_tests;
mod sql_update_postgres_tests;
mod sql_update_tests;
mod sql_write_tests;
mod sqlite_metadata_provider_test;
mod sqllogictest_runner;
mod table_changes_tests;
mod table_deletions_repartition_tests;
mod table_tests;
mod time_travel_tests;
mod type_promotion_tests;
mod view_tests;
mod write_inlining_tests;
mod write_tests;
