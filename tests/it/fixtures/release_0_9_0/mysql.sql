CREATE TABLE `ducklake_column` (
  `column_id` bigint DEFAULT NULL,
  `begin_snapshot` bigint DEFAULT NULL,
  `end_snapshot` bigint DEFAULT NULL,
  `table_id` bigint DEFAULT NULL,
  `column_order` bigint DEFAULT NULL,
  `column_name` varchar(1024) DEFAULT NULL,
  `column_type` varchar(1024) DEFAULT NULL,
  `initial_default` text,
  `default_value` text,
  `nulls_allowed` tinyint(1) DEFAULT NULL,
  `parent_column` bigint DEFAULT NULL,
  `default_value_type` varchar(1024) DEFAULT NULL,
  `default_value_dialect` varchar(1024) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
CREATE TABLE `ducklake_data_file` (
  `data_file_id` bigint NOT NULL AUTO_INCREMENT,
  `table_id` bigint NOT NULL,
  `path` text NOT NULL,
  `path_is_relative` tinyint(1) NOT NULL DEFAULT '1',
  `file_size_bytes` bigint NOT NULL,
  `footer_size` bigint DEFAULT NULL,
  `encryption_key` varchar(1024) DEFAULT NULL,
  `record_count` bigint DEFAULT NULL,
  `row_id_start` bigint DEFAULT NULL,
  `mapping_id` bigint DEFAULT NULL,
  `begin_snapshot` bigint NOT NULL,
  `end_snapshot` bigint DEFAULT NULL,
  `partial_max` bigint DEFAULT NULL,
  `partition_id` bigint DEFAULT NULL,
  PRIMARY KEY (`data_file_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
CREATE TABLE `ducklake_delete_file` (
  `delete_file_id` bigint NOT NULL AUTO_INCREMENT,
  `data_file_id` bigint NOT NULL,
  `table_id` bigint NOT NULL,
  `path` text NOT NULL,
  `path_is_relative` tinyint(1) NOT NULL DEFAULT '1',
  `file_size_bytes` bigint NOT NULL,
  `footer_size` bigint DEFAULT NULL,
  `encryption_key` varchar(1024) DEFAULT NULL,
  `delete_count` bigint DEFAULT NULL,
  `begin_snapshot` bigint NOT NULL,
  `end_snapshot` bigint DEFAULT NULL,
  `partial_max` bigint DEFAULT NULL,
  PRIMARY KEY (`delete_file_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
CREATE TABLE `ducklake_file_column_stats` (
  `data_file_id` bigint NOT NULL,
  `table_id` bigint NOT NULL,
  `column_id` bigint NOT NULL,
  `column_size_bytes` bigint DEFAULT NULL,
  `value_count` bigint DEFAULT NULL,
  `null_count` bigint DEFAULT NULL,
  `min_value` text,
  `max_value` text,
  `contains_nan` tinyint(1) DEFAULT NULL,
  `extra_stats` text
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
CREATE TABLE `ducklake_file_partition_value` (
  `data_file_id` bigint NOT NULL,
  `table_id` bigint NOT NULL,
  `partition_key_index` bigint NOT NULL,
  `partition_value` text,
  KEY `idx_file_partition_value_table_key` (`table_id`,`partition_key_index`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
CREATE TABLE `ducklake_files_scheduled_for_deletion` (
  `data_file_id` bigint NOT NULL,
  `path` text NOT NULL,
  `path_is_relative` tinyint(1) NOT NULL DEFAULT '1',
  `schedule_start` datetime(6) DEFAULT CURRENT_TIMESTAMP(6)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
CREATE TABLE `ducklake_inlined_data_tables` (
  `table_id` bigint NOT NULL,
  `table_name` varchar(1024) NOT NULL,
  `schema_version` bigint NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
CREATE TABLE `ducklake_metadata` (
  `key` varchar(1024) NOT NULL,
  `value` text NOT NULL,
  `scope` varchar(1024) DEFAULT NULL,
  `scope_id` bigint DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
INSERT INTO `ducklake_metadata` VALUES ('next_column_id','0',NULL,NULL),('next_snapshot_id','0',NULL,NULL),('next_file_id','0',NULL,NULL),('next_delete_file_id','0',NULL,NULL),('next_partition_id','0',NULL,NULL),('next_sort_id','0',NULL,NULL);
CREATE TABLE `ducklake_partition_column` (
  `partition_id` bigint NOT NULL,
  `table_id` bigint NOT NULL,
  `partition_key_index` bigint NOT NULL,
  `column_id` bigint NOT NULL,
  `transform` varchar(1024) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
CREATE TABLE `ducklake_partition_info` (
  `partition_id` bigint NOT NULL,
  `table_id` bigint NOT NULL,
  `begin_snapshot` bigint NOT NULL,
  `end_snapshot` bigint DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
CREATE TABLE `ducklake_schema` (
  `schema_id` bigint NOT NULL AUTO_INCREMENT,
  `schema_name` varchar(1024) NOT NULL,
  `path` text NOT NULL,
  `path_is_relative` tinyint(1) NOT NULL DEFAULT '1',
  `begin_snapshot` bigint NOT NULL,
  `end_snapshot` bigint DEFAULT NULL,
  PRIMARY KEY (`schema_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
CREATE TABLE `ducklake_schema_versions` (
  `begin_snapshot` bigint NOT NULL,
  `schema_version` bigint NOT NULL,
  `table_id` bigint NOT NULL,
  UNIQUE KEY `table_id` (`table_id`,`begin_snapshot`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
CREATE TABLE `ducklake_snapshot` (
  `snapshot_id` bigint NOT NULL,
  `snapshot_time` datetime(6) DEFAULT CURRENT_TIMESTAMP(6),
  `schema_version` bigint NOT NULL DEFAULT '0',
  PRIMARY KEY (`snapshot_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
CREATE TABLE `ducklake_snapshot_changes` (
  `snapshot_id` bigint NOT NULL,
  `changes_made` text,
  `author` text,
  `commit_message` text,
  `commit_extra_info` text,
  PRIMARY KEY (`snapshot_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
CREATE TABLE `ducklake_sort_expression` (
  `sort_id` bigint NOT NULL,
  `table_id` bigint NOT NULL,
  `sort_key_index` bigint NOT NULL,
  `expression` varchar(1024) NOT NULL,
  `dialect` varchar(256) NOT NULL,
  `sort_direction` varchar(16) NOT NULL,
  `null_order` varchar(16) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
CREATE TABLE `ducklake_sort_info` (
  `sort_id` bigint NOT NULL,
  `table_id` bigint NOT NULL,
  `begin_snapshot` bigint NOT NULL,
  `end_snapshot` bigint DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
CREATE TABLE `ducklake_table` (
  `table_id` bigint NOT NULL AUTO_INCREMENT,
  `schema_id` bigint NOT NULL,
  `table_name` varchar(1024) NOT NULL,
  `path` text NOT NULL,
  `path_is_relative` tinyint(1) NOT NULL DEFAULT '1',
  `begin_snapshot` bigint NOT NULL,
  `end_snapshot` bigint DEFAULT NULL,
  PRIMARY KEY (`table_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
CREATE TABLE `ducklake_table_column_stats` (
  `table_id` bigint NOT NULL,
  `column_id` bigint NOT NULL,
  `contains_null` tinyint(1) DEFAULT NULL,
  `contains_nan` tinyint(1) DEFAULT NULL,
  `min_value` text,
  `max_value` text,
  `extra_stats` text
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
CREATE TABLE `ducklake_table_stats` (
  `table_id` bigint NOT NULL,
  `record_count` bigint NOT NULL DEFAULT '0',
  `next_row_id` bigint NOT NULL DEFAULT '0',
  `file_size_bytes` bigint NOT NULL DEFAULT '0',
  PRIMARY KEY (`table_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
CREATE TABLE `ducklake_view` (
  `view_id` bigint DEFAULT NULL,
  `view_uuid` varchar(36) DEFAULT NULL,
  `begin_snapshot` bigint DEFAULT NULL,
  `end_snapshot` bigint DEFAULT NULL,
  `schema_id` bigint DEFAULT NULL,
  `view_name` varchar(1024) DEFAULT NULL,
  `dialect` varchar(1024) DEFAULT NULL,
  `sql` text,
  `column_aliases` text
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
