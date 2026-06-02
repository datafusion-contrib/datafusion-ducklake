//! High-level table writer for DuckLake catalogs.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::{Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStore, ObjectStoreExt, PutPayload};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use uuid::Uuid;

use crate::Result;
use crate::metadata_writer::{ColumnDef, DataFileInfo, MetadataWriter, WriteMode, WriteResult};
use crate::path_resolver::join_paths;

/// High-level writer for DuckLake tables.
#[derive(Debug)]
pub struct DuckLakeTableWriter {
    metadata: Arc<dyn MetadataWriter>,
    object_store: Arc<dyn ObjectStore>,
    /// The key path portion of the data_path (e.g., "/prefix/data/")
    base_key_path: String,
    /// Compression codec for written data files. Defaults to `UNCOMPRESSED`;
    /// override via [`DuckLakeTableWriter::with_compression`] to trade write
    /// CPU for ~2x smaller files (e.g. `LZ4`, `SNAPPY`, `ZSTD`).
    compression: Compression,
}

impl DuckLakeTableWriter {
    pub fn new(
        metadata: Arc<dyn MetadataWriter>,
        object_store: Arc<dyn ObjectStore>,
    ) -> Result<Self> {
        let data_path_str = metadata.get_data_path()?;
        let (_, key_path) = crate::path_resolver::parse_object_store_url(&data_path_str)?;

        Ok(Self {
            metadata,
            object_store,
            base_key_path: key_path,
            compression: Compression::UNCOMPRESSED,
        })
    }

    /// Override the parquet compression codec used for written data files.
    /// Defaults to [`Compression::UNCOMPRESSED`].
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Begin a streaming write session.
    /// If mode is `WriteMode::Replace`, ends existing files.
    pub fn begin_write(
        &self,
        schema_name: &str,
        table_name: &str,
        arrow_schema: &Schema,
        mode: WriteMode,
    ) -> Result<TableWriteSession> {
        let table_key = join_paths(&join_paths(&self.base_key_path, schema_name)?, table_name)?;
        let file_name = format!("{}.parquet", Uuid::new_v4());
        self.begin_write_internal(
            schema_name,
            table_name,
            arrow_schema,
            table_key,
            file_name.clone(),
            file_name,
            true,
            mode,
        )
    }

    /// Begin a streaming write session with a custom file path (registered as absolute).
    pub fn begin_write_to_path(
        &self,
        schema_name: &str,
        table_name: &str,
        arrow_schema: &Schema,
        file_dir: &str,
        file_name: String,
        mode: WriteMode,
    ) -> Result<TableWriteSession> {
        let full_path = join_paths(file_dir, &file_name)?;
        self.begin_write_internal(
            schema_name,
            table_name,
            arrow_schema,
            file_dir.to_string(),
            file_name,
            full_path,
            false,
            mode,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn begin_write_internal(
        &self,
        schema_name: &str,
        table_name: &str,
        arrow_schema: &Schema,
        file_dir: String,
        file_name: String,
        catalog_path: String,
        path_is_relative: bool,
        mode: WriteMode,
    ) -> Result<TableWriteSession> {
        let columns = arrow_schema_to_column_defs(arrow_schema)?;
        let setup =
            self.metadata
                .begin_write_transaction(schema_name, table_name, &columns, mode)?;
        let schema_with_ids =
            Arc::new(build_schema_with_field_ids(arrow_schema, &setup.column_ids));

        let object_path_str = join_paths(&file_dir, &file_name)?;
        // Strip leading slash for object_store Path (it expects relative keys)
        let object_path = ObjectPath::from(object_path_str.trim_start_matches('/'));

        let props = WriterProperties::builder()
            .set_writer_version(parquet::file::properties::WriterVersion::PARQUET_2_0)
            .set_compression(self.compression)
            .build();
        let writer = ArrowWriter::try_new(Vec::new(), schema_with_ids.clone(), Some(props))?;

        Ok(TableWriteSession {
            metadata: Arc::clone(&self.metadata),
            object_store: Arc::clone(&self.object_store),
            object_path,
            snapshot_id: setup.snapshot_id,
            schema_id: setup.schema_id,
            table_id: setup.table_id,
            column_ids: setup.column_ids,
            schema_with_ids,
            writer: Some(writer),
            catalog_path,
            path_is_relative,
            row_count: 0,
        })
    }

    /// Write batches to a table, replacing any existing data.
    pub async fn write_table(
        &self,
        schema_name: &str,
        table_name: &str,
        batches: &[RecordBatch],
    ) -> Result<WriteResult> {
        if batches.is_empty() {
            return Err(crate::error::DuckLakeError::InvalidConfig(
                "No batches to write".to_string(),
            ));
        }

        let arrow_schema = batches[0].schema();
        let mut session =
            self.begin_write(schema_name, table_name, &arrow_schema, WriteMode::Replace)?;

        for batch in batches {
            session.write_batch(batch)?;
        }

        session.finish().await
    }

    /// Write batches to a table, appending to existing data.
    pub async fn append_table(
        &self,
        schema_name: &str,
        table_name: &str,
        batches: &[RecordBatch],
    ) -> Result<WriteResult> {
        if batches.is_empty() {
            return Err(crate::error::DuckLakeError::InvalidConfig(
                "No batches to write".to_string(),
            ));
        }

        let arrow_schema = batches[0].schema();
        let mut session =
            self.begin_write(schema_name, table_name, &arrow_schema, WriteMode::Append)?;

        for batch in batches {
            session.write_batch(batch)?;
        }

        session.finish().await
    }
}

/// Streaming write session. Buffer is dropped if not finished (no data uploaded).
#[derive(Debug)]
pub struct TableWriteSession {
    metadata: Arc<dyn MetadataWriter>,
    object_store: Arc<dyn ObjectStore>,
    object_path: ObjectPath,
    snapshot_id: i64,
    schema_id: i64,
    table_id: i64,
    #[allow(dead_code)]
    column_ids: Vec<i64>,
    schema_with_ids: SchemaRef,
    writer: Option<ArrowWriter<Vec<u8>>>,
    /// Path to register in catalog (may be relative filename or absolute path)
    catalog_path: String,
    /// Whether the catalog_path is relative to table path
    path_is_relative: bool,
    row_count: i64,
}

impl TableWriteSession {
    pub fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        if self.writer.is_none() {
            return Err(crate::error::DuckLakeError::Internal(
                "Writer already closed".to_string(),
            ));
        }
        self.validate_batch_schema(batch)?;

        let batch_with_ids =
            RecordBatch::try_new(self.schema_with_ids.clone(), batch.columns().to_vec())?;
        let writer = self.writer.as_mut().unwrap();
        writer.write(&batch_with_ids)?;
        self.row_count += batch.num_rows() as i64;
        Ok(())
    }

    fn validate_batch_schema(&self, batch: &RecordBatch) -> Result<()> {
        let batch_schema = batch.schema();
        let expected_schema = &self.schema_with_ids;

        if batch_schema.fields().len() != expected_schema.fields().len() {
            return Err(crate::error::DuckLakeError::InvalidConfig(format!(
                "Schema mismatch: batch has {} columns, expected {}",
                batch_schema.fields().len(),
                expected_schema.fields().len()
            )));
        }

        for (i, (batch_field, expected_field)) in batch_schema
            .fields()
            .iter()
            .zip(expected_schema.fields().iter())
            .enumerate()
        {
            if batch_field.data_type() != expected_field.data_type() {
                return Err(crate::error::DuckLakeError::InvalidConfig(format!(
                    "Schema mismatch at column {}: batch has type {:?}, expected {:?}",
                    i,
                    batch_field.data_type(),
                    expected_field.data_type()
                )));
            }
        }
        Ok(())
    }

    pub fn row_count(&self) -> i64 {
        self.row_count
    }

    pub fn snapshot_id(&self) -> i64 {
        self.snapshot_id
    }

    /// Returns the object path that will be written to
    pub fn file_path(&self) -> &str {
        self.object_path.as_ref()
    }

    pub async fn finish(mut self) -> Result<WriteResult> {
        let writer = self.writer.take().ok_or_else(|| {
            crate::error::DuckLakeError::Internal("Writer already closed".to_string())
        })?;
        let buffer = writer.into_inner()?;

        let file_size = buffer.len() as i64;
        let footer_size = calculate_footer_size_from_bytes(&buffer)?;

        // Upload via object_store
        self.object_store
            .put(&self.object_path, PutPayload::from(buffer))
            .await?;

        let mut file_info = DataFileInfo::new(&self.catalog_path, file_size, self.row_count)
            .with_footer_size(footer_size);
        if !self.path_is_relative {
            file_info = file_info.with_absolute_path();
        }
        self.metadata
            .register_data_file(self.table_id, self.snapshot_id, &file_info)?;

        Ok(WriteResult {
            snapshot_id: self.snapshot_id,
            table_id: self.table_id,
            schema_id: self.schema_id,
            files_written: 1,
            records_written: self.row_count,
        })
    }
}

// Drop is a no-op: buffer is simply dropped, nothing was uploaded to the store.

fn arrow_schema_to_column_defs(schema: &Schema) -> Result<Vec<ColumnDef>> {
    schema
        .fields()
        .iter()
        .map(|field| ColumnDef::from_arrow(field.name(), field.data_type(), field.is_nullable()))
        .collect()
}

fn build_schema_with_field_ids(schema: &Schema, column_ids: &[i64]) -> Schema {
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .zip(column_ids.iter())
        .map(|(field, &col_id)| {
            let mut metadata: HashMap<String, String> = field.metadata().clone();
            metadata.insert("PARQUET:field_id".to_string(), col_id.to_string());
            Field::new(field.name(), field.data_type().clone(), field.is_nullable())
                .with_metadata(metadata)
        })
        .collect();

    Schema::new_with_metadata(fields, schema.metadata().clone())
}

fn calculate_footer_size_from_bytes(buffer: &[u8]) -> Result<i64> {
    if buffer.len() < 8 {
        return Err(crate::error::DuckLakeError::Internal(
            "Invalid Parquet file: too small".to_string(),
        ));
    }

    let footer_bytes = &buffer[buffer.len() - 8..];

    if &footer_bytes[4..8] != b"PAR1" {
        return Err(crate::error::DuckLakeError::Internal(
            "Invalid Parquet file: missing PAR1 magic".to_string(),
        ));
    }

    let metadata_len =
        i32::from_le_bytes([footer_bytes[0], footer_bytes[1], footer_bytes[2], footer_bytes[3]])
            as i64;
    Ok(metadata_len + 8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::DataType;

    #[test]
    fn test_arrow_schema_to_column_defs() {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]);

        let columns = arrow_schema_to_column_defs(&schema).unwrap();
        assert_eq!(columns.len(), 2);
        assert_eq!(columns[0].name, "id");
        assert_eq!(columns[0].ducklake_type, "int32");
        assert!(!columns[0].is_nullable);
        assert_eq!(columns[1].name, "name");
        assert_eq!(columns[1].ducklake_type, "varchar");
        assert!(columns[1].is_nullable);
    }

    #[test]
    fn test_build_schema_with_field_ids() {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]);

        let column_ids = vec![1, 2];
        let schema_with_ids = build_schema_with_field_ids(&schema, &column_ids);

        // Check that field_ids are embedded in metadata
        let field0_metadata = schema_with_ids.field(0).metadata();
        assert_eq!(
            field0_metadata.get("PARQUET:field_id"),
            Some(&"1".to_string())
        );

        let field1_metadata = schema_with_ids.field(1).metadata();
        assert_eq!(
            field1_metadata.get("PARQUET:field_id"),
            Some(&"2".to_string())
        );
    }

    #[test]
    fn test_write_parquet_to_buffer_with_field_ids() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap();

        let column_ids = vec![10, 20];
        let schema_with_ids = Arc::new(build_schema_with_field_ids(&schema, &column_ids));

        let props = WriterProperties::builder()
            .set_writer_version(parquet::file::properties::WriterVersion::PARQUET_2_0)
            .build();
        let mut writer =
            ArrowWriter::try_new(Vec::new(), schema_with_ids.clone(), Some(props)).unwrap();

        let batch_with_ids =
            RecordBatch::try_new(schema_with_ids, batch.columns().to_vec()).unwrap();
        writer.write(&batch_with_ids).unwrap();
        let buffer = writer.into_inner().unwrap();

        let file_size = buffer.len() as i64;
        let footer_size = calculate_footer_size_from_bytes(&buffer).unwrap();

        assert!(file_size > 0);
        assert!(footer_size > 0);
        assert!(footer_size < file_size);
    }

    #[test]
    fn test_calculate_footer_size_from_bytes() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();

        let props = WriterProperties::builder()
            .set_writer_version(parquet::file::properties::WriterVersion::PARQUET_2_0)
            .build();
        let schema_with_ids = Arc::new(build_schema_with_field_ids(&batch.schema(), &[1]));
        let mut writer =
            ArrowWriter::try_new(Vec::new(), schema_with_ids.clone(), Some(props)).unwrap();

        let batch_with_ids =
            RecordBatch::try_new(schema_with_ids, batch.columns().to_vec()).unwrap();
        writer.write(&batch_with_ids).unwrap();
        let buffer = writer.into_inner().unwrap();

        let footer_size = calculate_footer_size_from_bytes(&buffer).unwrap();

        // Footer should be reasonable size (metadata + 8 bytes)
        assert!(footer_size >= 8);
        assert!(footer_size < 10000);
    }
}
