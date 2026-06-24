//! High-level table writer for DuckLake catalogs.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

use arrow::datatypes::{Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use object_store::ObjectStore;
use object_store::buffered::BufWriter as ObjectBufWriter;
use object_store::path::Path as ObjectPath;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::NamedTempFile;
use tokio::io::AsyncWriteExt;
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
    /// Optional max rows per parquet row group. `None` leaves the parquet
    /// default. Set via [`DuckLakeTableWriter::with_max_row_group_rows`].
    max_row_group_rows: Option<usize>,
    /// Optional max *uncompressed* bytes per parquet row group. `None` leaves
    /// the parquet default (rows-only). A reader decodes a whole row group at
    /// once, so a byte cap bounds reader memory for wide schemas (e.g. large
    /// vector columns). Set via [`DuckLakeTableWriter::with_max_row_group_bytes`].
    max_row_group_bytes: Option<usize>,
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
            max_row_group_rows: None,
            max_row_group_bytes: None,
        })
    }

    /// Override the parquet compression codec used for written data files.
    /// Defaults to [`Compression::UNCOMPRESSED`].
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Cap the number of rows per parquet row group. Leaves the parquet
    /// default when unset.
    pub fn with_max_row_group_rows(mut self, rows: usize) -> Self {
        self.max_row_group_rows = Some(rows);
        self
    }

    /// Cap the *uncompressed* bytes per parquet row group, flushing the row
    /// group once it is reached. Because a parquet reader must decode an entire
    /// row group into memory at once, this bounds reader memory for wide
    /// schemas (e.g. large `List`/`FixedSizeList` vector columns) that would
    /// otherwise build multi-GiB row groups at the rows-only default. Leaves
    /// the parquet default when unset.
    pub fn with_max_row_group_bytes(mut self, bytes: usize) -> Self {
        self.max_row_group_bytes = Some(bytes);
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
        // Multicatalog backends share one physical `data_path`, so without a
        // per-catalog segment two catalogs writing the same (schema, table)
        // would dump files into the same directory. Prepend `cat_{id}` to keep
        // them physically isolated. Single-catalog backends report `None` and
        // skip the segment, preserving the historical `{schema}/{table}/…`
        // layout. `cat_` prefix + numeric id is rename-safe and needs no
        // sanitisation.
        let scoped_base = match self.metadata.catalog_id() {
            Some(id) => join_paths(&self.base_key_path, &format!("cat_{id}"))?,
            None => self.base_key_path.clone(),
        };
        let table_key = join_paths(&join_paths(&scoped_base, schema_name)?, table_name)?;
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

        // Apply caller-configured row-group caps. The ArrowWriter enforces both
        // natively (flushing the row group when either is hit). The byte cap
        // matters for wide schemas: a parquet reader decodes a whole row group
        // at once, so an uncapped large vector column builds multi-GiB row
        // groups that OOM readers. Both default to the parquet default (unset).
        let mut props_builder = WriterProperties::builder()
            .set_writer_version(parquet::file::properties::WriterVersion::PARQUET_2_0)
            .set_compression(self.compression);
        if let Some(rows) = self.max_row_group_rows {
            props_builder = props_builder.set_max_row_group_row_count(Some(rows));
        }
        if let Some(bytes) = self.max_row_group_bytes {
            props_builder = props_builder.set_max_row_group_bytes(Some(bytes));
        }
        let props = props_builder.build();
        // Stream the parquet to a local staging file rather than an in-memory
        // buffer: a multi-GB table would otherwise be held whole in RAM and,
        // worse, uploaded as a single PUT (object stores cap a single PUT at
        // 5 GiB). `finish()` streams this file out via a multipart upload.
        let temp = NamedTempFile::new()?;
        let staging = std::io::BufWriter::new(temp.reopen()?);
        let writer = ArrowWriter::try_new(staging, schema_with_ids.clone(), Some(props))?;

        Ok(TableWriteSession {
            metadata: Arc::clone(&self.metadata),
            object_store: Arc::clone(&self.object_store),
            object_path,
            schema_name: schema_name.to_string(),
            table_name: table_name.to_string(),
            snapshot_id: setup.snapshot_id,
            base_snapshot_id: setup.base_snapshot_id,
            table_id: setup.table_id,
            columns,
            column_ids: setup.column_ids,
            schema_with_ids,
            writer: Some(writer),
            temp: Some(temp),
            catalog_path,
            path_is_relative,
            mode,
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

/// Streaming write session. Batches stream to a local staging file; the
/// finished parquet is uploaded in `finish()`. If the session is dropped
/// without finishing, the staging file is removed and nothing is uploaded.
#[derive(Debug)]
pub struct TableWriteSession {
    metadata: Arc<dyn MetadataWriter>,
    object_store: Arc<dyn ObjectStore>,
    object_path: ObjectPath,
    /// Target identifiers threaded to `register_data_file`. Multicatalog Postgres
    /// writes the schema/table metadata at the commit (keyed by these names);
    /// single-catalog SQLite ignores them (it created them at begin).
    schema_name: String,
    table_name: String,
    snapshot_id: i64,
    /// Catalog head observed at `begin_write_transaction`; threaded to
    /// `register_data_file` so a `Replace` commit can abort if another writer
    /// published a newer generation of the table since this write began.
    base_snapshot_id: i64,
    table_id: i64,
    /// Column generation for this write (in `column_order`). Threaded to the
    /// metadata writer at `finish()` so single-catalog backends, which defer the
    /// column generation out of `begin_write_transaction`, can insert the
    /// column rows with `column_ids` at the atomic commit.
    columns: Vec<ColumnDef>,
    column_ids: Vec<i64>,
    schema_with_ids: SchemaRef,
    /// Parquet writer streaming to the local staging file (`temp`). Batches are
    /// written to disk as they arrive rather than buffered in memory, so peak
    /// memory stays bounded by the parquet row-group size regardless of table
    /// size. The finished file is streamed to object storage in `finish()`.
    writer: Option<ArrowWriter<std::io::BufWriter<std::fs::File>>>,
    /// Local staging file backing `writer`. Kept alive for the session; the
    /// finished parquet is uploaded from it and the file is removed on drop.
    temp: Option<NamedTempFile>,
    /// Path to register in catalog (may be relative filename or absolute path)
    catalog_path: String,
    /// Whether the catalog_path is relative to table path
    path_is_relative: bool,
    /// Replace vs Append; passed to `register_data_file` so the head advance and
    /// (for Replace) prior-generation retirement commit atomically with the file.
    mode: WriteMode,
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
        let temp = self.temp.take().ok_or_else(|| {
            crate::error::DuckLakeError::Internal("Writer already closed".to_string())
        })?;

        // Finalise the parquet footer, then unwrap the `BufWriter` (its
        // `into_inner` flushes any buffered footer bytes to the OS file) so the
        // staging file on disk is the complete parquet.
        let staged = writer.into_inner()?;
        let mut file = staged
            .into_inner()
            .map_err(|e| crate::error::DuckLakeError::Io(e.into_error()))?;

        let file_size = file.metadata()?.len() as i64;
        let footer_size = read_footer_size(&mut file)?;

        // Stream the staged file to object storage. `BufWriter` chunks the
        // payload and switches to a multipart upload for large files, so there
        // is no 5 GiB single-PUT ceiling and memory stays bounded. On failure
        // we abort so no incomplete multipart parts are left behind.
        let local = tokio::fs::File::open(temp.path()).await?;
        let mut reader = tokio::io::BufReader::new(local);
        let mut upload =
            ObjectBufWriter::new(Arc::clone(&self.object_store), self.object_path.clone());
        if let Err(e) = stream_to_upload(&mut reader, &mut upload).await {
            let _ = upload.abort().await;
            return Err(e.into());
        }

        let mut file_info = DataFileInfo::new(&self.catalog_path, file_size, self.row_count)
            .with_footer_size(footer_size);
        if !self.path_is_relative {
            file_info = file_info.with_absolute_path();
        }
        // register_data_file returns the ids actually committed (snapshot id
        // assigned at commit; real schema/table ids, which may differ from the
        // begin-time reservations under a concurrent create). Report those.
        let committed = self.metadata.register_data_file(
            self.table_id,
            &self.schema_name,
            &self.table_name,
            self.snapshot_id,
            &file_info,
            self.mode,
            self.base_snapshot_id,
            &self.columns,
            &self.column_ids,
        )?;

        Ok(WriteResult {
            snapshot_id: committed.snapshot_id,
            table_id: committed.table_id,
            schema_id: committed.schema_id,
            files_written: 1,
            records_written: self.row_count,
        })
    }
}

// Drop deletes the staging `NamedTempFile`; a session abandoned before
// `finish()` uploads nothing and leaves no local file behind.

/// Stream a finished local parquet file to object storage and finalise the
/// upload. `BufWriter` switches to a multipart upload once the payload exceeds
/// its buffer, so files larger than the object store's single-PUT limit (5 GiB
/// on S3) upload fine and memory stays bounded.
async fn stream_to_upload<R>(reader: &mut R, upload: &mut ObjectBufWriter) -> std::io::Result<()>
where
    R: tokio::io::AsyncRead + Unpin + ?Sized,
{
    tokio::io::copy(reader, upload).await?;
    upload.shutdown().await?;
    Ok(())
}

/// Read the parquet footer length (thrift metadata + 8-byte trailer) from the
/// tail of a finished parquet file on disk. Stored as the nullable
/// `footer_size` hint in the catalog; readers fall back to a standard footer
/// read when it is absent.
fn read_footer_size(file: &mut std::fs::File) -> Result<i64> {
    let len = file.metadata()?.len();
    if len < 8 {
        return Err(crate::error::DuckLakeError::Internal(
            "Invalid Parquet file: too small".to_string(),
        ));
    }
    file.seek(SeekFrom::End(-8))?;
    let mut tail = [0u8; 8];
    file.read_exact(&mut tail)?;
    calculate_footer_size_from_bytes(&tail)
}

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
