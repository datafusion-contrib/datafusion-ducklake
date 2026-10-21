PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE ducklake_inlined_data_tables (
    table_id INTEGER,
    table_name VARCHAR,
    schema_version INTEGER
);
CREATE TABLE ducklake_tag (
    object_id BIGINT,
    begin_snapshot BIGINT,
    end_snapshot BIGINT,
    key VARCHAR,
    value VARCHAR
);
CREATE TABLE ducklake_column_tag (
    table_id BIGINT,
    column_id BIGINT,
    begin_snapshot BIGINT,
    end_snapshot BIGINT,
    key VARCHAR,
    value VARCHAR
);
CREATE TABLE ducklake_metadata (
    key VARCHAR NOT NULL,
    value VARCHAR NOT NULL,
    scope VARCHAR,
    scope_id INTEGER
);
INSERT INTO ducklake_metadata VALUES('next_column_id','0',NULL,NULL);
INSERT INTO ducklake_metadata VALUES('next_partition_id','0',NULL,NULL);
INSERT INTO ducklake_metadata VALUES('next_sort_id','0',NULL,NULL);
CREATE TABLE ducklake_snapshot (
    snapshot_id INTEGER PRIMARY KEY,
    snapshot_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    schema_version INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE ducklake_schema_versions (
    begin_snapshot INTEGER NOT NULL,
    schema_version INTEGER NOT NULL,
    table_id INTEGER NOT NULL,
    UNIQUE (table_id, begin_snapshot)
);
CREATE TABLE ducklake_schema (
    schema_id INTEGER PRIMARY KEY,
    schema_name VARCHAR NOT NULL,
    path VARCHAR NOT NULL DEFAULT '',
    path_is_relative BOOLEAN NOT NULL DEFAULT 1,
    begin_snapshot INTEGER NOT NULL,
    end_snapshot INTEGER
);
CREATE TABLE ducklake_table (
    table_id INTEGER PRIMARY KEY,
    schema_id INTEGER NOT NULL,
    table_name VARCHAR NOT NULL,
    path VARCHAR NOT NULL DEFAULT '',
    path_is_relative BOOLEAN NOT NULL DEFAULT 1,
    begin_snapshot INTEGER NOT NULL,
    end_snapshot INTEGER
);
CREATE TABLE ducklake_view (
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
CREATE TABLE ducklake_column (
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
CREATE TABLE ducklake_data_file (
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
CREATE TABLE ducklake_snapshot_changes (
    snapshot_id INTEGER PRIMARY KEY,
    changes_made VARCHAR,
    author VARCHAR,
    commit_message VARCHAR,
    commit_extra_info VARCHAR
);
CREATE TABLE ducklake_table_stats (
    table_id INTEGER PRIMARY KEY,
    record_count INTEGER NOT NULL DEFAULT 0,
    next_row_id INTEGER NOT NULL DEFAULT 0,
    file_size_bytes INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE ducklake_file_column_stats (
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
CREATE TABLE ducklake_table_column_stats (
    table_id INTEGER NOT NULL,
    column_id INTEGER NOT NULL,
    contains_null BOOLEAN,
    contains_nan BOOLEAN,
    min_value VARCHAR,
    max_value VARCHAR,
    extra_stats VARCHAR
);
CREATE TABLE ducklake_delete_file (
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
CREATE TABLE ducklake_files_scheduled_for_deletion (
    data_file_id INTEGER NOT NULL,
    path VARCHAR NOT NULL,
    path_is_relative BOOLEAN NOT NULL DEFAULT 1,
    schedule_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE ducklake_partition_info (
    partition_id INTEGER NOT NULL,
    table_id INTEGER NOT NULL,
    begin_snapshot INTEGER NOT NULL,
    end_snapshot INTEGER
);
CREATE TABLE ducklake_partition_column (
    partition_id INTEGER NOT NULL,
    table_id INTEGER NOT NULL,
    partition_key_index INTEGER NOT NULL,
    column_id INTEGER NOT NULL,
    transform VARCHAR NOT NULL
);
CREATE TABLE ducklake_file_partition_value (
    data_file_id INTEGER NOT NULL,
    table_id INTEGER NOT NULL,
    partition_key_index INTEGER NOT NULL,
    partition_value VARCHAR
);
CREATE TABLE ducklake_sort_info (
    sort_id INTEGER NOT NULL,
    table_id INTEGER NOT NULL,
    begin_snapshot INTEGER NOT NULL,
    end_snapshot INTEGER
);
CREATE TABLE ducklake_sort_expression (
    sort_id INTEGER NOT NULL,
    table_id INTEGER NOT NULL,
    sort_key_index INTEGER NOT NULL,
    expression VARCHAR NOT NULL,
    dialect VARCHAR NOT NULL,
    sort_direction VARCHAR NOT NULL,
    null_order VARCHAR NOT NULL
);
CREATE INDEX idx_file_partition_value_table_key
    ON ducklake_file_partition_value (table_id, partition_key_index);
COMMIT;
