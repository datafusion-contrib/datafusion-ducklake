//! DuckDB file provenance and safe compaction.

#![cfg(all(feature = "write-duckdb", feature = "metadata-duckdb"))]

use std::any::Any;
use std::sync::Arc;

use arrow::array::{Int32Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::SessionContext;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTable, DuckLakeTableWriter, DuckdbMetadataProvider,
    DuckdbMetadataWriter, MergeOptions, MetadataProvider, MetadataWriter,
};
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

#[tokio::test]
async fn duckdb_compaction_retains_origin_metadata_and_rows() {
    let temp = TempDir::new().unwrap();
    let database = temp.path().join("catalog.ducklake");
    let writer = Arc::new(DuckdbMetadataWriter::new_with_init(database.to_str().unwrap()).unwrap());
    writer
        .set_data_path(temp.path().join("data").to_str().unwrap())
        .unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let table_writer =
        DuckLakeTableWriter::new(writer.clone(), Arc::new(LocalFileSystem::new())).unwrap();
    let mut snapshots = Vec::new();
    let mut table_id = 0;
    for index in 0..3 {
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![index * 2, index * 2 + 1]))],
        )
        .unwrap();
        let result = if index == 0 {
            table_writer
                .write_table("main", "events", &[batch])
                .await
                .unwrap()
        } else {
            table_writer
                .append_table("main", "events", &[batch])
                .await
                .unwrap()
        };
        table_id = result.table_id;
        snapshots.push(result.snapshot_id);
    }
    let provider = Arc::new(DuckdbMetadataProvider::new(database.to_string_lossy()).unwrap());
    let files = provider
        .get_table_files_for_select(table_id, *snapshots.last().unwrap())
        .unwrap();
    assert_eq!(
        files
            .iter()
            .map(|file| file.begin_snapshot)
            .collect::<Vec<_>>(),
        snapshots.iter().copied().map(Some).collect::<Vec<_>>()
    );
    assert_eq!(
        files
            .iter()
            .map(|file| file.schema_version)
            .collect::<Vec<_>>(),
        vec![Some(1); 3]
    );

    let catalog = DuckLakeCatalog::with_writer(provider.clone(), writer).unwrap();
    let context = SessionContext::new();
    context.register_catalog("lake", Arc::new(catalog));
    let table_provider = context
        .catalog("lake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("events")
        .await
        .unwrap()
        .unwrap();
    let table = (table_provider.as_ref() as &dyn Any)
        .downcast_ref::<DuckLakeTable>()
        .unwrap();
    let options = MergeOptions {
        target_file_size: 1 << 20,
        ..Default::default()
    };
    let result = table
        .merge_adjacent_files(&context.state(), options)
        .await
        .unwrap();
    assert_eq!(
        (
            result.files_processed,
            result.files_created,
            result.rows_written
        ),
        (3, 1, 6)
    );
    let fresh = Arc::new(DuckdbMetadataProvider::new(database.to_string_lossy()).unwrap());
    let current = fresh.get_current_snapshot().unwrap();
    let files = fresh.get_table_files_for_select(table_id, current).unwrap();
    assert_eq!(current, snapshots.last().unwrap() + 1);
    assert_eq!(files.len(), 1);
    assert_eq!(
        (
            files[0].begin_snapshot,
            files[0].schema_version,
            files[0].partial_max
        ),
        (Some(snapshots[0]), Some(1), snapshots.last().copied()),
    );
    assert_eq!(
        read_rows(DuckLakeCatalog::with_snapshot(fresh.clone(), current).unwrap()).await,
        vec![(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
    );
    assert_eq!(
        read_rows(DuckLakeCatalog::with_snapshot(fresh.clone(), snapshots[0]).unwrap()).await,
        vec![(0, 0), (1, 1)]
    );
    assert_eq!(
        read_rows(DuckLakeCatalog::with_snapshot(fresh, snapshots[1]).unwrap()).await,
        vec![(0, 0), (1, 1), (2, 2), (3, 3)]
    );
}

async fn read_rows(catalog: DuckLakeCatalog) -> Vec<(i32, i64)> {
    let context = SessionContext::new();
    context.register_catalog("lake", Arc::new(catalog.with_row_lineage(true)));
    let batches = context
        .sql("SELECT id, rowid FROM lake.main.events ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut rows = Vec::new();
    for batch in batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let rowids = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        rows.extend((0..batch.num_rows()).map(|index| (ids.value(index), rowids.value(index))));
    }
    rows
}
