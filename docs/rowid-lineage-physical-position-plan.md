# Plan: row-id lineage and deletes must use physical file position

Status: proposed
Scope: `src/table.rs`, `src/row_id.rs`, `src/delete_filter.rs`, tests
DataFusion version: 53.0.0

---

## Problem

DuckLake row lineage defines:

```text
rowid = row_id_start + physical_row_position
```

`physical_row_position` is the row's 0-based position in the physical Parquet file. Positional
delete files use the same position in their `pos` column.

The current implementation computes that position with a per-stream counter:

- `RowIdExec` starts `cursor = 0` for the stream and appends `row_id_start + cursor + i`.
- `DeleteFilterExec` starts `row_offset = 0` for the stream and checks `row_offset + i`.

That only works when one file is read by one ordered stream. DataFusion can split one Parquet file
into multiple byte-range partitions. `RowIdExec` then forces an unordered `CoalescePartitionsExec`
because it requires `SinglePartition`, and `DeleteFilterExec` runs once per split partition with its
counter reset to 0. Both produce silent wrong answers on large files.

Confirmed repro:

- One file from `range(0, 600000)`, `row_id_start = 0`.
- `target_partitions = 8`, `repartition_file_min_size = 1`.
- `SELECT rowid, i ...` produces mismatches such as `(rowid=0, i=245760)`.
- `DELETE WHERE i IN [245760, 245770)` leaves those deleted rows visible.

This is a correctness bug, not a performance bug.

---

## Design Goal

Match DuckDB/DuckLake semantics: derive positions from Parquet physical row order, not from
DataFusion arrival order.

DataFusion 53 does not expose a reader-level `file_row_number` column like DuckDB, so the pragmatic
fix is:

1. Control the scan shape for positional paths.
2. Emit a physical-position column immediately above the scan.
3. Make rowid synthesis and delete filtering read that column instead of counting stream rows.

The key invariant is:

```text
FileRowNumberExec is correct only if its input partition emits a complete, contiguous, in-order run
of physical rows.
```

Everything below enforces that invariant.

---

## Resolved Decisions

- Use a fixed operator order for the first implementation:
  `DataSourceExec -> FileRowNumberExec -> DeleteFilterExec -> RowIdExec -> final projection`.
- Do not rely on `RowIdExec` and `DeleteFilterExec` being freely reorderable.
- Do not apply `LIMIT` inside per-file positional plans.
- If an internal limit is ever needed, apply it once above the combined table plan, not once per file.
- Refuse reader-side filter, sort/reverse, and byte-range repartitioning on positional paths.
- Preserve row counts when dropping the internal position column to a zero-column output.

---

## Affected Paths

Use the positional path when either condition is true:

- `rowid` is projected.
- The file has a positional delete file.

Unaffected path:

- Files without deletes when `rowid` is not projected keep the existing grouped Parquet scan and
  scan-level limit/pushdown behavior.

---

## Plan Shape

For each affected data file:

```text
Final projection / rename       (drops __ducklake_row_pos, preserves row count for zero columns)
  RowIdExec                     (if rowid projected; reads row_pos, appends rowid, passes row_pos)
    DeleteFilterExec            (if deletes; reads row_pos)
      FileRowNumberExec         (appends __ducklake_row_pos)
        DataSourceExec          (row-group chunks, no byte re-split, no reader pruning/reordering)
```

DataFusion's outer plan remains responsible for global query filters, sorts, and limits.

---

## Implementation

### 1. Regression Tests First

Add tests that fail on the current code:

- Large split file: `SELECT rowid, i` must satisfy `rowid == i` when `row_id_start = 0`.
- Large split file with deletes: deleted physical positions must not appear.
- Large split file with both rowid and deletes.
- Filtered rowid query, for example `WHERE i >= K LIMIT 1`, must return the correct rowid.
- `ORDER BY` on a positional path must not corrupt rowids.
- `LIMIT 1` with the first physical row deleted must return the next survivor, not zero rows.
- `COUNT(*)` with deletes must preserve row counts through the internal position column.
- Single-row-group file remains correct.
- Multi-file table remains correct.
- Embedded-rowid file with deletes remains correct.
- `row_id_start = None`: rowid projection hard-errors as today; delete filtering still works.

### 2. Read Row-Group Metadata

Extend `FileReadConfig` with:

```rust
row_group_starts: Vec<i64>,
row_group_count: usize,
```

Populate these from the `ParquetMetaData` already opened in `build_file_read_config`:

```rust
row_group_starts[0] = 0
row_group_starts[i + 1] = row_group_starts[i] + row_groups[i].num_rows()
```

The catalog does not store per-row-group counts; the Parquet footer is the source of truth.

### 3. Build Row-Group Partitions

Add a helper:

```rust
fn build_row_group_partitions(
    file: &DuckLakeFileData,
    read_cfg: &FileReadConfig,
    target_partitions: usize,
) -> DataFusionResult<(Vec<FileGroup>, Vec<i64>)>
```

Rules:

- Split row groups into `min(target_partitions, row_group_count)` contiguous chunks.
- Balance chunks by row count where practical.
- For each chunk `[a, b)`:
  - Create a `ParquetAccessPlan` of length `row_group_count`.
  - Mark row groups in `[a, b)` as `Scan`; all others as `Skip`.
  - Do not use `Selection(RowSelection)`.
  - Attach the plan to a `PartitionedFile` with the same path, file size, and footer-size hint.
  - Wrap that one `PartitionedFile` in its own `FileGroup`.
  - Push `row_group_starts[a]` into `partition_starts`.

DataFusion partitions are `FileGroup`s, not files. The returned vectors must be 1:1:

```text
file_groups[i] <-> partition_starts[i]
```

### 4. Wrap ParquetSource For Positional Scans

Create a small `FileSource` wrapper around `ParquetSource` for positional paths.

It must refuse anything that can change row order, row membership, or partition layout:

- `supports_repartitioning() -> false`
- `repartitioned(...) -> Ok(None)`
- `try_pushdown_filters(...)`: return all filters as not pushed.
- `try_pushdown_sort(...)`: return unsupported.
- `try_reverse_output(...)`: return unsupported.

It may delegate and re-wrap order/cardinality-preserving source-returning methods:

- `with_batch_size`
- `with_schema_adapter_factory`
- `try_pushdown_projection`

Read-only accessors delegate normally:

- `create_file_opener`
- `as_any`
- `table_schema`
- `filter`
- `projection`
- `metrics`
- `file_type`
- `fmt_extra`
- `schema_adapter_factory`

This wrapper is not an optimization layer. It exists to keep `FileRowNumberExec`'s input as complete,
contiguous physical rows in physical order.

### 5. Add FileRowNumberExec

Add `FileRowNumberExec` in `src/row_pos.rs` or `src/row_id.rs`.

Behavior:

- Input: row-group-partitioned `DataSourceExec`.
- State: `partition_starts: Arc<Vec<i64>>`.
- Output: input columns plus internal `Int64` column `__ducklake_row_pos`.
- `execute(partition)` starts at `partition_starts[partition]`.
- For each batch, append values `start + cursor + i`, then increment `cursor` by `batch.num_rows()`.

Plan properties:

- Preserve input partitioning.
- Preserve input order.
- Do not require `SinglePartition`.

Add an EXPLAIN assertion that no coalesce/repartition/filter/sort appears between the scan and
`FileRowNumberExec` on positional paths.

### 6. Rewrite Consumers

`DeleteFilterExec`:

- Stop using `row_offset`.
- Find `__ducklake_row_pos`.
- Keep rows whose position is not in the delete set.
- Preserve the internal row-position column in its output.
- Preserve zero-column row counts where applicable.

`RowIdExec`:

- Stop using `cursor`.
- Remove `required_input_distribution = SinglePartition`.
- Find `__ducklake_row_pos`.
- Append `rowid = row_id_start + __ducklake_row_pos`.
- Pass `__ducklake_row_pos` through unchanged.
- Keep current behavior for missing `row_id_start`: non-embedded files with projected rowid hard-error
  in `table.rs` before execution.

Embedded rowid files:

- Continue reading the embedded `_ducklake_internal_row_id` column for rowid.
- If the file also has deletes, still add `FileRowNumberExec` so delete filtering uses physical
  positions.

### 7. Wire table.rs

For `build_exec_for_file_with_rowid`:

- If the file needs synthetic row positions, use row-group partitions and the positional Parquet
  source.
- Pass `None` to the scan builder's limit.
- Build:

```text
DataSourceExec
  -> FileRowNumberExec
  -> DeleteFilterExec if deletes
  -> RowIdExec if synthetic rowid is needed
  -> final projection/rename
```

For embedded-rowid files with projected rowid:

```text
DataSourceExec
  -> FileRowNumberExec if deletes
  -> DeleteFilterExec if deletes
  -> final projection/rename
```

For `build_exec_for_file_with_deletes`:

- Use the same row-group partitioning and positional source.
- Pass `None` to the scan builder's limit.
- Build:

```text
DataSourceExec
  -> FileRowNumberExec
  -> DeleteFilterExec
  -> final projection/rename
```

For `build_exec_for_files_without_deletes`:

- Leave unchanged.

Limit handling:

- Per-file positional builders must receive/use `None` for scan limit.
- Do not add a per-file `GlobalLimitExec`.
- Rely on DataFusion's outer limit, or apply one limit once above `combine_execution_plans` if that
  becomes necessary.

Final projection:

- Drop `__ducklake_row_pos` before exposing the catalog schema.
- If dropping to zero output columns, create the output `RecordBatch` with
  `RecordBatchOptions::with_row_count(Some(batch.num_rows()))`.

---

## Non-Goals

- Do not implement a custom Parquet opener in this fix.
- Do not preserve reader-side row-group/page/bloom pruning on positional paths.
- Do not push sort/reverse/filters into the reader on positional paths.
- Do not refactor unrelated scan paths.

Future improvement: compute a reader-level file-row-number column, DuckDB-style, to recover full
reader pruning and ordering optimizations.

---

## Verification

Run:

- New large split-file rowid tests.
- New large split-file delete tests.
- Existing row lineage tests.
- Existing delete tests.
- Full test suite.

Plan assertions for positional paths:

- No `CoalescePartitionsExec` inserted below `FileRowNumberExec`.
- No reader predicate on the positional `DataSourceExec`.
- No sort/reverse pushdown in the positional source.
- No scan-level limit on positional `DataSourceExec`.
- One `FileGroup` per row-group chunk.

Correctness assertions:

- `rowid == row_id_start + physical_position`.
- Deleted physical positions are absent.
- `LIMIT` after deletes returns survivors, not pre-delete rows.
- `COUNT(*)` with deletes returns the correct count.
