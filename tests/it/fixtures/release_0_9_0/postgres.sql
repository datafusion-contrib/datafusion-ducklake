CREATE TABLE public.ducklake_column (
    column_id bigint,
    begin_snapshot bigint,
    end_snapshot bigint,
    table_id bigint,
    column_order bigint,
    column_name character varying,
    column_type character varying,
    initial_default character varying,
    default_value character varying,
    nulls_allowed boolean,
    parent_column bigint,
    default_value_type character varying,
    default_value_dialect character varying
);
CREATE TABLE public.ducklake_column_tag (
    table_id bigint,
    column_id bigint,
    begin_snapshot bigint,
    end_snapshot bigint,
    key character varying,
    value character varying
);
CREATE TABLE public.ducklake_data_file (
    data_file_id bigint NOT NULL,
    table_id bigint NOT NULL,
    path character varying NOT NULL,
    path_is_relative boolean DEFAULT true NOT NULL,
    file_size_bytes bigint NOT NULL,
    footer_size bigint,
    encryption_key character varying,
    record_count bigint,
    row_id_start bigint,
    mapping_id bigint,
    begin_snapshot bigint NOT NULL,
    end_snapshot bigint,
    partition_id bigint
);
ALTER TABLE public.ducklake_data_file ALTER COLUMN data_file_id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.ducklake_data_file_data_file_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);
CREATE TABLE public.ducklake_delete_file (
    delete_file_id bigint NOT NULL,
    data_file_id bigint NOT NULL,
    table_id bigint NOT NULL,
    path character varying NOT NULL,
    path_is_relative boolean DEFAULT true NOT NULL,
    file_size_bytes bigint NOT NULL,
    footer_size bigint,
    encryption_key character varying,
    delete_count bigint,
    begin_snapshot bigint NOT NULL,
    end_snapshot bigint
);
ALTER TABLE public.ducklake_delete_file ALTER COLUMN delete_file_id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.ducklake_delete_file_delete_file_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);
CREATE TABLE public.ducklake_file_column_stats (
    data_file_id bigint NOT NULL,
    table_id bigint NOT NULL,
    column_id bigint NOT NULL,
    column_size_bytes bigint,
    value_count bigint,
    null_count bigint,
    min_value character varying,
    max_value character varying,
    contains_nan boolean,
    extra_stats character varying
);
CREATE TABLE public.ducklake_file_partition_value (
    data_file_id bigint NOT NULL,
    table_id bigint NOT NULL,
    partition_key_index bigint NOT NULL,
    partition_value character varying
);
CREATE TABLE public.ducklake_files_scheduled_for_deletion (
    data_file_id bigint NOT NULL,
    path character varying NOT NULL,
    path_is_relative boolean DEFAULT true NOT NULL,
    schedule_start timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE public.ducklake_inlined_data_tables (
    table_id bigint,
    table_name character varying,
    schema_version bigint
);
CREATE TABLE public.ducklake_metadata (
    key character varying NOT NULL,
    value character varying NOT NULL,
    scope character varying,
    scope_id bigint
);
CREATE TABLE public.ducklake_partition_column (
    partition_id bigint NOT NULL,
    table_id bigint NOT NULL,
    partition_key_index bigint NOT NULL,
    column_id bigint NOT NULL,
    transform character varying NOT NULL
);
CREATE TABLE public.ducklake_partition_info (
    partition_id bigint NOT NULL,
    table_id bigint NOT NULL,
    begin_snapshot bigint NOT NULL,
    end_snapshot bigint
);
CREATE TABLE public.ducklake_schema (
    schema_id bigint NOT NULL,
    schema_name character varying NOT NULL,
    path character varying DEFAULT ''::character varying NOT NULL,
    path_is_relative boolean DEFAULT true NOT NULL,
    begin_snapshot bigint NOT NULL,
    end_snapshot bigint
);
ALTER TABLE public.ducklake_schema ALTER COLUMN schema_id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.ducklake_schema_schema_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);
CREATE TABLE public.ducklake_schema_versions (
    begin_snapshot bigint NOT NULL,
    schema_version bigint NOT NULL,
    table_id bigint NOT NULL
);
CREATE TABLE public.ducklake_snapshot (
    snapshot_id bigint NOT NULL,
    snapshot_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    schema_version bigint DEFAULT 0 NOT NULL
);
CREATE TABLE public.ducklake_snapshot_changes (
    snapshot_id bigint NOT NULL,
    changes_made character varying,
    author character varying,
    commit_message character varying,
    commit_extra_info character varying
);
CREATE TABLE public.ducklake_sort_expression (
    sort_id bigint NOT NULL,
    table_id bigint NOT NULL,
    sort_key_index bigint NOT NULL,
    expression character varying NOT NULL,
    dialect character varying NOT NULL,
    sort_direction character varying NOT NULL,
    null_order character varying NOT NULL
);
CREATE TABLE public.ducklake_sort_info (
    sort_id bigint NOT NULL,
    table_id bigint NOT NULL,
    begin_snapshot bigint NOT NULL,
    end_snapshot bigint
);
CREATE TABLE public.ducklake_table (
    table_id bigint NOT NULL,
    schema_id bigint NOT NULL,
    table_name character varying NOT NULL,
    path character varying DEFAULT ''::character varying NOT NULL,
    path_is_relative boolean DEFAULT true NOT NULL,
    begin_snapshot bigint NOT NULL,
    end_snapshot bigint
);
CREATE TABLE public.ducklake_table_column_stats (
    table_id bigint NOT NULL,
    column_id bigint NOT NULL,
    contains_null boolean,
    contains_nan boolean,
    min_value character varying,
    max_value character varying,
    extra_stats character varying
);
CREATE TABLE public.ducklake_table_stats (
    table_id bigint NOT NULL,
    record_count bigint DEFAULT 0 NOT NULL,
    next_row_id bigint DEFAULT 0 NOT NULL,
    file_size_bytes bigint DEFAULT 0 NOT NULL
);
ALTER TABLE public.ducklake_table ALTER COLUMN table_id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.ducklake_table_table_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);
CREATE TABLE public.ducklake_tag (
    object_id bigint,
    begin_snapshot bigint,
    end_snapshot bigint,
    key character varying,
    value character varying
);
CREATE TABLE public.ducklake_view (
    view_id bigint,
    view_uuid uuid,
    begin_snapshot bigint,
    end_snapshot bigint,
    schema_id bigint,
    view_name character varying,
    dialect character varying,
    sql character varying,
    column_aliases character varying
);
INSERT INTO public.ducklake_metadata VALUES ('next_column_id', '0', NULL, NULL);
INSERT INTO public.ducklake_metadata VALUES ('next_snapshot_id', '0', NULL, NULL);
INSERT INTO public.ducklake_metadata VALUES ('next_partition_id', '0', NULL, NULL);
INSERT INTO public.ducklake_metadata VALUES ('next_sort_id', '0', NULL, NULL);
SELECT pg_catalog.setval('public.ducklake_data_file_data_file_id_seq', 1, false);
SELECT pg_catalog.setval('public.ducklake_delete_file_delete_file_id_seq', 1, false);
SELECT pg_catalog.setval('public.ducklake_schema_schema_id_seq', 1, false);
SELECT pg_catalog.setval('public.ducklake_table_table_id_seq', 1, false);
ALTER TABLE ONLY public.ducklake_data_file
    ADD CONSTRAINT ducklake_data_file_pkey PRIMARY KEY (data_file_id);
ALTER TABLE ONLY public.ducklake_delete_file
    ADD CONSTRAINT ducklake_delete_file_pkey PRIMARY KEY (delete_file_id);
ALTER TABLE ONLY public.ducklake_schema
    ADD CONSTRAINT ducklake_schema_pkey PRIMARY KEY (schema_id);
ALTER TABLE ONLY public.ducklake_schema_versions
    ADD CONSTRAINT ducklake_schema_versions_table_id_begin_snapshot_key UNIQUE (table_id, begin_snapshot);
ALTER TABLE ONLY public.ducklake_snapshot_changes
    ADD CONSTRAINT ducklake_snapshot_changes_pkey PRIMARY KEY (snapshot_id);
ALTER TABLE ONLY public.ducklake_snapshot
    ADD CONSTRAINT ducklake_snapshot_pkey PRIMARY KEY (snapshot_id);
ALTER TABLE ONLY public.ducklake_table
    ADD CONSTRAINT ducklake_table_pkey PRIMARY KEY (table_id);
ALTER TABLE ONLY public.ducklake_table_stats
    ADD CONSTRAINT ducklake_table_stats_pkey PRIMARY KEY (table_id);
CREATE INDEX idx_file_partition_value_table_key ON public.ducklake_file_partition_value USING btree (table_id, partition_key_index);
