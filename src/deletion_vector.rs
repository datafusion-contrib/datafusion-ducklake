//! Iceberg V3 deletion-vector and Puffin encoding and decoding.

use std::collections::{BTreeMap, HashSet};
use std::io::Cursor;

use arrow::array::{Int64Array, RecordBatch, StringArray};
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext, object_store::ObjectStoreUrl};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use object_store::{ObjectStore, ObjectStoreExt, path::Path as ObjectPath};
use std::sync::Arc;

use roaring::RoaringBitmap;
use serde_json::Value;

use crate::{DuckLakeError, Result};

const DELETION_VECTOR_MAGIC: [u8; 4] = [0xD1, 0xD3, 0x39, 0x64];
const PUFFIN_MAGIC: &[u8; 4] = b"PFA1";
const PUFFIN_FOOTER_TAIL_SIZE: usize = 12;
const PUFFIN_MIN_FILE_SIZE: usize = 20;
const DELETION_VECTOR_BLOB_TYPE: &str = "deletion-vector-v1";
const DUCKLAKE_SNAPSHOT_PROPERTY: &str = "ducklake-snapshot-id";

#[derive(Debug, Clone, Copy)]
struct PuffinBlob {
    offset: usize,
    length: usize,
    snapshot_id: Option<i64>,
}

pub(crate) fn decode_deletion_entries(bytes: &[u8], path: &str) -> Result<Vec<(i64, Option<i64>)>> {
    let blobs = parse_blobs(bytes, path)?;
    let Some(first) = blobs.first() else {
        return corrupt(path, "contains no deletion-vector-v1 blobs");
    };
    let with_snapshot = first.snapshot_id.is_some();
    let mut positions: BTreeMap<i64, Option<i64>> = BTreeMap::new();
    for blob in blobs {
        if blob.snapshot_id.is_some() != with_snapshot {
            return corrupt(path, "mixes blobs with and without snapshot IDs");
        }
        let mut decoded = HashSet::new();
        decode_blob(
            &bytes[blob.offset..blob.offset + blob.length],
            path,
            &mut decoded,
        )?;
        for position in decoded {
            positions
                .entry(position)
                .and_modify(|snapshot| *snapshot = (*snapshot).min(blob.snapshot_id))
                .or_insert(blob.snapshot_id);
        }
    }
    Ok(positions.into_iter().collect())
}

#[cfg(test)]
fn decode_deletion_vectors(bytes: &[u8], snapshot_id: i64, path: &str) -> Result<HashSet<i64>> {
    Ok(decode_deletion_entries(bytes, path)?
        .into_iter()
        .filter(|(_, snapshot)| snapshot.is_none_or(|value| value <= snapshot_id))
        .map(|(position, _)| position)
        .collect())
}

pub(crate) fn encode_deletion_vector(positions: &[i64]) -> Result<Vec<u8>> {
    let mut groups = BTreeMap::<i32, RoaringBitmap>::new();
    for &position in positions {
        if position < 0 {
            return Err(DuckLakeError::InvalidConfig(format!(
                "Deletion vector position must be non-negative, got {position}"
            )));
        }
        let position = u64::try_from(position).unwrap();
        let key = i32::try_from(position >> 32).unwrap();
        groups.entry(key).or_default().insert(position as u32);
    }

    let mut payload = DELETION_VECTOR_MAGIC.to_vec();
    payload.extend_from_slice(&i64::try_from(groups.len()).unwrap().to_le_bytes());
    for (key, bitmap) in groups {
        payload.extend_from_slice(&key.to_le_bytes());
        bitmap.serialize_into(&mut payload).map_err(|e| {
            DuckLakeError::InvalidConfig(format!(
                "Failed to encode deletion vector Roaring bitmap: {e}"
            ))
        })?;
    }

    let payload_size = u32::try_from(payload.len()).map_err(|_| {
        DuckLakeError::InvalidConfig("Deletion vector exceeds the 4 GiB format limit".to_string())
    })?;
    let checksum = crc32fast::hash(&payload);
    let mut result = payload_size.to_be_bytes().to_vec();
    result.extend_from_slice(&payload);
    result.extend_from_slice(&checksum.to_be_bytes());
    Ok(result)
}

fn parse_blobs(bytes: &[u8], path: &str) -> Result<Vec<PuffinBlob>> {
    if bytes.len() >= 8 && bytes[4..8] == DELETION_VECTOR_MAGIC {
        return Ok(vec![PuffinBlob {
            offset: 0,
            length: bytes.len(),
            snapshot_id: None,
        }]);
    }
    if bytes.len() < PUFFIN_MIN_FILE_SIZE || &bytes[..4] != PUFFIN_MAGIC {
        return corrupt(path, "has no deletion-vector or Puffin magic");
    }
    if &bytes[bytes.len() - 4..] != PUFFIN_MAGIC {
        return corrupt(path, "has invalid trailing Puffin magic");
    }

    let payload_size = read_u32_le(bytes, bytes.len() - PUFFIN_FOOTER_TAIL_SIZE)? as usize;
    let flags = read_u32_le(bytes, bytes.len() - 8)?;
    if flags != 0 {
        return corrupt(path, "uses unsupported Puffin footer flags");
    }
    let Some(payload_start) = bytes
        .len()
        .checked_sub(PUFFIN_FOOTER_TAIL_SIZE + payload_size)
    else {
        return corrupt(path, "has an out-of-range Puffin footer size");
    };
    if payload_start < 8 || &bytes[payload_start - 4..payload_start] != PUFFIN_MAGIC {
        return corrupt(path, "has invalid Puffin footer magic");
    }
    let blob_section_end = payload_start - 4;
    parse_footer_blobs(
        &bytes[payload_start..payload_start + payload_size],
        blob_section_end,
        path,
    )
}

fn parse_footer_blobs(
    bytes: &[u8],
    blob_section_end: usize,
    path: &str,
) -> Result<Vec<PuffinBlob>> {
    let footer: Value = serde_json::from_slice(bytes).map_err(|e| {
        DuckLakeError::InvalidConfig(format!(
            "Deletion vector file '{path}' has invalid Puffin footer JSON: {e}"
        ))
    })?;
    let blobs = footer
        .get("blobs")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            DuckLakeError::InvalidConfig(format!(
                "Deletion vector file '{path}' has no Puffin blobs list"
            ))
        })?;

    blobs
        .iter()
        .filter(|blob| blob.get("type").and_then(Value::as_str) == Some(DELETION_VECTOR_BLOB_TYPE))
        .map(|blob| parse_blob_metadata(blob, blob_section_end, path))
        .collect()
}

fn parse_blob_metadata(blob: &Value, blob_section_end: usize, path: &str) -> Result<PuffinBlob> {
    let offset = json_usize(blob, "offset", path)?;
    let length = json_usize(blob, "length", path)?;
    if offset < 4
        || length == 0
        || offset
            .checked_add(length)
            .is_none_or(|end| end > blob_section_end)
    {
        return corrupt(path, "has out-of-range Puffin blob metadata");
    }
    let snapshot_id = blob
        .get("properties")
        .and_then(|properties| properties.get(DUCKLAKE_SNAPSHOT_PROPERTY))
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| {
                    DuckLakeError::InvalidConfig(format!(
                        "Deletion vector file '{path}' has a non-string snapshot ID"
                    ))
                })?
                .parse::<i64>()
                .map_err(|e| {
                    DuckLakeError::InvalidConfig(format!(
                        "Deletion vector file '{path}' has an invalid snapshot ID: {e}"
                    ))
                })
        })
        .transpose()?;
    if snapshot_id.is_some_and(|snapshot| snapshot < 0) {
        return corrupt(path, "has a negative snapshot ID");
    }
    Ok(PuffinBlob {
        offset,
        length,
        snapshot_id,
    })
}

fn decode_blob(blob: &[u8], path: &str, positions: &mut HashSet<i64>) -> Result<()> {
    if blob.len() < 20 {
        return corrupt(path, "contains a deletion vector shorter than 20 bytes");
    }
    let vector_size = u32::from_be_bytes(blob[..4].try_into().unwrap()) as usize;
    if vector_size.checked_add(8) != Some(blob.len()) {
        return corrupt(
            path,
            "contains a deletion vector with an invalid length prefix",
        );
    }
    if blob[4..8] != DELETION_VECTOR_MAGIC {
        return corrupt(path, "contains invalid deletion-vector-v1 magic");
    }
    let bitmap_count = i64::from_le_bytes(blob[8..16].try_into().unwrap());
    if bitmap_count < 0 {
        return corrupt(path, "contains a negative Roaring bitmap count");
    }
    let checksum_offset = blob.len() - 4;
    let expected_checksum = u32::from_be_bytes(blob[checksum_offset..].try_into().unwrap());
    let actual_checksum = crc32fast::hash(&blob[4..checksum_offset]);
    if actual_checksum != expected_checksum {
        return corrupt(path, "contains a deletion vector with an invalid CRC-32");
    }

    let mut offset = 16;
    let mut keys = HashSet::new();
    for _ in 0..bitmap_count {
        if offset + 4 > checksum_offset {
            return corrupt(path, "ends before a Roaring bitmap key");
        }
        let key = i32::from_le_bytes(blob[offset..offset + 4].try_into().unwrap());
        offset += 4;
        if key < 0 || !keys.insert(key) {
            return corrupt(path, "contains an invalid or duplicate Roaring bitmap key");
        }
        let mut cursor = Cursor::new(&blob[offset..checksum_offset]);
        let bitmap = RoaringBitmap::deserialize_from(&mut cursor).map_err(|e| {
            DuckLakeError::InvalidConfig(format!(
                "Deletion vector file '{path}' contains an invalid Roaring bitmap: {e}"
            ))
        })?;
        let consumed = usize::try_from(cursor.position()).unwrap();
        if consumed == 0 {
            return corrupt(path, "contains an empty Roaring bitmap encoding");
        }
        offset += consumed;
        for low in bitmap {
            let position = (u64::try_from(key).unwrap() << 32) | u64::from(low);
            positions.insert(i64::try_from(position).unwrap());
        }
    }
    if offset != checksum_offset {
        return corrupt(path, "contains trailing deletion-vector payload bytes");
    }
    Ok(())
}

fn json_usize(value: &Value, field: &str, path: &str) -> Result<usize> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            DuckLakeError::InvalidConfig(format!(
                "Deletion vector file '{path}' has invalid Puffin blob {field}"
            ))
        })
}

fn read_u32_le(bytes: &[u8], offset: usize) -> Result<u32> {
    bytes
        .get(offset..offset + 4)
        .and_then(|value| value.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| DuckLakeError::InvalidConfig("Truncated Puffin footer".to_string()))
}

fn corrupt<T>(path: &str, message: &str) -> Result<T> {
    Err(DuckLakeError::InvalidConfig(format!(
        "Deletion vector file '{path}' {message}"
    )))
}

pub(crate) async fn read_deletion_entries(
    store: &dyn ObjectStore,
    path: &str,
) -> Result<Vec<(i64, Option<i64>)>> {
    let bytes = store
        .get(&ObjectPath::from(path.trim_start_matches('/')))
        .await?
        .bytes()
        .await?;
    decode_deletion_entries(&bytes, path)
}

pub(crate) async fn deletion_vector_has_snapshots(
    store: &dyn ObjectStore,
    path: &str,
    size: i64,
) -> Result<bool> {
    let size = u64::try_from(size).map_err(|_| {
        DuckLakeError::InvalidConfig(format!("Invalid deletion vector size for '{path}'"))
    })?;
    if size < PUFFIN_MIN_FILE_SIZE as u64 {
        return corrupt(path, "is shorter than a deletion vector");
    }
    let object = ObjectPath::from(path.trim_start_matches('/'));
    let head = store.get_range(&object, 0..8).await?;
    if head.get(4..8) == Some(DELETION_VECTOR_MAGIC.as_slice()) {
        return Ok(false);
    }
    if head.get(..4) != Some(PUFFIN_MAGIC.as_slice()) {
        return corrupt(path, "has no deletion-vector or Puffin magic");
    }
    let tail_start = size - PUFFIN_FOOTER_TAIL_SIZE as u64;
    let tail = store.get_range(&object, tail_start..size).await?;
    if tail.get(8..12) != Some(PUFFIN_MAGIC.as_slice()) {
        return corrupt(path, "has invalid trailing Puffin magic");
    }
    if read_u32_le(&tail, 4)? != 0 {
        return corrupt(path, "uses unsupported Puffin footer flags");
    }
    let length = u64::from(read_u32_le(&tail, 0)?);
    let Some(start) = tail_start
        .checked_sub(length + 4)
        .filter(|start| *start >= 4)
    else {
        return corrupt(path, "has an out-of-range Puffin footer size");
    };
    let footer = store.get_range(&object, start..tail_start).await?;
    if footer.get(..4) != Some(PUFFIN_MAGIC.as_slice()) {
        return corrupt(path, "has invalid Puffin footer magic");
    }
    let end = usize::try_from(start).map_err(|_| {
        DuckLakeError::InvalidConfig(format!("Invalid Puffin footer offset for '{path}'"))
    })?;
    let blobs = parse_footer_blobs(&footer[4..], end, path)?;
    Ok(blobs.iter().any(|blob| blob.snapshot_id.is_some()))
}

#[derive(Debug)]
pub(crate) struct DeletionVectorExec {
    path: String,
    url: ObjectStoreUrl,
    properties: Arc<PlanProperties>,
}

impl DeletionVectorExec {
    pub(crate) fn new(path: String, url: ObjectStoreUrl) -> Arc<Self> {
        let schema = crate::table::delete_file_read_schema();
        Arc::new(Self {
            path,
            url,
            properties: Arc::new(PlanProperties::new(
                EquivalenceProperties::new(schema),
                Partitioning::UnknownPartitioning(1),
                EmissionType::Final,
                Boundedness::Bounded,
            )),
        })
    }
}

impl DisplayAs for DeletionVectorExec {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DeletionVectorExec")
    }
}

impl ExecutionPlan for DeletionVectorExec {
    fn name(&self) -> &str {
        "DeletionVectorExec"
    }

    fn apply_expressions(
        &self,
        _: &mut dyn FnMut(
            &Arc<dyn datafusion::physical_expr::PhysicalExpr>,
        )
            -> DataFusionResult<datafusion::common::tree_node::TreeNodeRecursion>,
    ) -> DataFusionResult<datafusion::common::tree_node::TreeNodeRecursion> {
        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(DataFusionError::Internal(
                "DeletionVectorExec has no children".to_string(),
            ));
        }
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "DeletionVectorExec has one partition, got {partition}"
            )));
        }
        let store = context.runtime_env().object_store(&self.url)?;
        let path = self.path.clone();
        let schema = self.schema();
        let batch_schema = Arc::clone(&schema);
        let stream = futures::stream::once(async move {
            let entries = read_deletion_entries(store.as_ref(), &path).await?;
            let (positions, snapshots): (Vec<_>, Vec<_>) = entries.into_iter().unzip();
            let paths = StringArray::from(vec![""; positions.len()]);
            Ok(RecordBatch::try_new(
                batch_schema,
                vec![
                    Arc::new(paths),
                    Arc::new(Int64Array::from(positions)),
                    Arc::new(Int64Array::from(snapshots)),
                ],
            )?)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn blob(groups: &[(i32, &[u32])]) -> Vec<u8> {
        let mut payload = DELETION_VECTOR_MAGIC.to_vec();
        payload.extend_from_slice(&(groups.len() as i64).to_le_bytes());
        for (key, values) in groups {
            payload.extend_from_slice(&key.to_le_bytes());
            let bitmap = values.iter().copied().collect::<RoaringBitmap>();
            bitmap.serialize_into(&mut payload).unwrap();
        }
        let checksum = crc32fast::hash(&payload);
        let mut result = u32::try_from(payload.len()).unwrap().to_be_bytes().to_vec();
        result.extend_from_slice(&payload);
        result.extend_from_slice(&checksum.to_be_bytes());
        result
    }

    fn puffin(blobs: &[(i64, Vec<u8>)]) -> Vec<u8> {
        let mut result = PUFFIN_MAGIC.to_vec();
        let mut metadata = Vec::new();
        for (snapshot_id, blob) in blobs {
            let offset = result.len();
            result.extend_from_slice(blob);
            metadata.push(json!({
                "type": DELETION_VECTOR_BLOB_TYPE,
                "offset": offset,
                "length": blob.len(),
                "properties": {DUCKLAKE_SNAPSHOT_PROPERTY: snapshot_id.to_string()}
            }));
        }
        result.extend_from_slice(PUFFIN_MAGIC);
        let footer = serde_json::to_vec(&json!({"blobs": metadata})).unwrap();
        result.extend_from_slice(&footer);
        result.extend_from_slice(&u32::try_from(footer.len()).unwrap().to_le_bytes());
        result.extend_from_slice(&0_u32.to_le_bytes());
        result.extend_from_slice(PUFFIN_MAGIC);
        result
    }

    #[test]
    fn decodes_bare_64_bit_positions() {
        let bytes = blob(&[(0, &[1, 7]), (1, &[3])]);
        assert_eq!(
            decode_deletion_vectors(&bytes, 9, "bare.puffin").unwrap(),
            HashSet::from([1, 7, 4_294_967_299])
        );
    }

    #[test]
    fn selects_cumulative_blobs_by_snapshot() {
        let bytes = puffin(&[(3, blob(&[(0, &[1, 2])])), (5, blob(&[(0, &[1, 2, 8])]))]);
        assert_eq!(
            decode_deletion_vectors(&bytes, 4, "history.puffin").unwrap(),
            HashSet::from([1, 2])
        );
        assert_eq!(
            decode_deletion_vectors(&bytes, 5, "history.puffin").unwrap(),
            HashSet::from([1, 2, 8])
        );
    }

    #[test]
    fn rejects_corrupt_checksum() {
        let mut bytes = blob(&[(0, &[1])]);
        *bytes.last_mut().unwrap() ^= 1;
        let error = decode_deletion_vectors(&bytes, 1, "bad.puffin").unwrap_err();
        assert!(error.to_string().contains("invalid CRC-32"));
    }

    #[test]
    fn encodes_bare_64_bit_positions() {
        let positions = [0, 7, 4_294_967_299, i64::MAX];
        let bytes = encode_deletion_vector(&positions).unwrap();
        assert_eq!(
            decode_deletion_vectors(&bytes, 1, "written.puffin").unwrap(),
            HashSet::from(positions)
        );
    }

    #[test]
    fn rejects_negative_write_position() {
        let error = encode_deletion_vector(&[-1]).unwrap_err();
        assert!(error.to_string().contains("non-negative"));
    }
}
