# Physical row positions in DuckLake reads

Status: current
Scope: `src/row_id.rs`, `src/delete_filter.rs`, `src/table.rs`, `src/table_changes.rs`,
`src/table_deletions.rs`
DataFusion version: 55.0.0

---

## What needs a position

DuckLake defines row lineage as:

```text
rowid = row_id_start + physical_row_position
```

`physical_row_position` is the row's 0-based position in the physical Parquet file. Positional
delete files use the same position in their `pos` column, so two features need it:

- **Positional deletes.** A delete file `(file_path VARCHAR, pos BIGINT)` names rows to drop by
  physical position; merge-on-read must drop exactly those.
- **Row lineage.** `rowid` for a file written by INSERT. (Files written by `UPDATE` or compaction
  instead embed their original rowids as a Parquet column tagged with the reserved field-id
  `2147483540`, and do not need a position.)

A scan that needs either is a **positional path**.

## Where the position comes from

The Parquet reader produces it, as a virtual column.

`row_id::positional_table_schema` builds the scan's `TableSchema` with one extra field carrying
arrow-rs's `RowNumber` extension type (`parquet.virtual.row_number`). DataFusion's Parquet opener
forwards that field to `ArrowReaderOptions::with_virtual_columns`, and the reader fills it from each
row group's absolute first-row offset in the footer.

Because the values come from footer offsets rather than from counting rows as they arrive, they stay
true physical positions under every transformation a scan may undergo:

- row-group pruning by statistics,
- page-index and bloom-filter pruning,
- row-level `RowSelection` from a pushed predicate,
- byte-range splitting of one file across partitions,
- reverse-order row-group reads.

This matches official DuckLake, which computes `rowid` from DuckDB's reader-level
`COLUMN_IDENTIFIER_FILE_ROW_NUMBER` virtual column
(`ducklake_multi_file_reader.cpp::GetVirtualColumnExpression`).

## Column layout

The position column is appended **last** in the scan's table schema, after the file columns. Every
positional call site therefore appends its table-schema index to its projection, so the position
lands last in the scan's output batches too.

Consumers take that index explicitly rather than looking the column up by name:

- `DeleteFilterExec::try_new(input, file_path, deleted_positions, pos_index)`
- `RowIdExec::try_new(input, row_id_start, pos_index)`

Two consumers derive the index arithmetically instead — `table_deletions.rs` and `table_changes.rs`
compute `table_len + embedded_rowid? + embedded_snapshot?`. Those sites carry a `debug_assert!` that
the arithmetic agrees with the scan's actual last column. All the internal columns are `Int64`, so a
misalignment would not fail a downcast; it would return wrong rowids.

### Why not look it up by name

`ROW_POS_COLUMN_NAME` (`__ducklake_row_pos`) is **not reserved**. DuckLake places no restriction on
column names and official DuckLake validates none, so a table may legitimately have a column called
`__ducklake_row_pos`. `row_id::unique_row_pos_name` suffixes the internal name until it does not
collide with the file's own columns — the same disambiguation DataFusion applies to its internal
row-index column. A name lookup would bind the user's column instead.

The position column is never written to a Parquet file and never appears in the catalog. It exists
only between the scan and its consumers, and `ColumnRenameExec` drops it before the table's output
schema.

## Filter pushdown on positional paths

Absolute positions make pruning safe: `filter(delete(R)) == delete(filter(R))`, because deletion is
keyed by position and a predicate is row-local, so dropping non-matching rows in the reader cannot
change which surviving row sits at which position.

Three nodes forward pushdown so a predicate can reach the reader:

- `DeleteFilterExec` — output schema equals input schema; forwards everything unchanged.
- `RowIdExec` — appends `rowid` last; forwards filters over input columns, rejects filters on
  `rowid` itself (DataFusion refuses any pushed predicate referencing a virtual column, and `rowid`
  is derived from one).
- `ColumnRenameExec` — forwards under `is_type_preserving_projection()`, which permits dropping the
  position column. It rewrites column *names*; `ChildFilterDescription::from_child` re-resolves
  indices by name.

### The NaN barrier

Parquet footer bounds for float columns exclude NaN, while DataFusion evaluates `NaN > C` as true
(IEEE 754 totalOrder, matching DuckDB). A predicate reaching the reader can therefore prune a row
group whose recorded max is below `C` but which still holds matching NaN rows.

`NanPruningBarrierExec` blocks predicates referencing a float column whose NaN state is not known
false for every scanned file. **Every path that can let a predicate reach the reader needs it.**
Current placement:

```
build_exec_for_files_without_deletes    barrier
build_exec_for_file_with_rowid          barrier
build_exec_for_file_with_deletes        barrier
build_exec_for_partial_file             none needed - see below
```

`build_exec_for_partial_file` has no barrier because `SnapshotFilterExec` does not implement
`gather_filters_for_pushdown`, so DataFusion's default bars every parent filter and no predicate
reaches the reader. **Give `SnapshotFilterExec` filter pushdown and that path needs a barrier too.**

### `resolve_positions` (the DELETE/UPDATE write path)

This path collects its plan directly and evaluates the predicate itself, so it has no pushdown
negotiation and no barrier node above it. It pushes the predicate into the reader only when
`predicate_is_prunable` allows, which is **stricter** than the read path on two counts:

- **Any float reference disqualifies the predicate**, not merely one whose file is not known
  NaN-free. Here a row the reader hides is not a missing result but a row a `DELETE` silently fails
  to delete. The read path can afford the precise `contains_nan` test because it has the catalog's
  per-file statistics; this path is handed a bare `DuckLakeFileData` and has none. Plumbing them
  through would let float predicates prune here too.
- **A file carrying any physical rename disqualifies every predicate.** The predicate holds catalog
  column names and the reader resolves a pushed predicate by name against the file's schema, which
  uses physical names. There is no `ColumnRenameExec` on this path to rewrite them. Pushing anyway
  is not merely slow: the unresolvable column becomes a null literal, the predicate folds to false,
  every row group prunes, and the `DELETE` removes nothing.

Both rejections fall back to scanning the whole file — what this path always did — never to a wrong
answer. `positional_pushdown_tests` pins both by disabling the guard and observing a `DELETE` that
removes zero rows.

## History

Before DataFusion 55 there was no reader-level row-number column. The crate synthesized positions
with a `FileRowNumberExec` above a deliberately constrained scan: row-group-aligned partitions with
plan-time seeds, wrapped in a `PositionalFileSource` that refused repartitioning, filter pushdown and
sort pushdown, because any of those would have shifted the positions it was counting. That machinery
is gone; `FileRowNumberExec` and `PositionalFileSource` no longer exist.

Note what that design did and did not cost. It **did** parallelize — `build_row_group_partitions`
split a file into `min(target_partitions, row_group_count)` chunks by hand — so byte-range splitting
is a simplification, not a new capability. What it could not do was **prune**: a measured A/B over a
5M-row DuckDB-written file (41 row groups) shows selective `rowid` queries 2.4-3x faster, while a
full scan with no prunable predicate is unchanged.

The parallelism substitution holds only on **read** paths, which run through the physical optimizer
where `repartition_file_scans` splits a file group. `build_update_scan_with_snapshot` and the
compaction sources are executed directly (`physical_plan::collect`, `input.execute(0, ..)`), so no
optimizer rule ever sees them: a single-file `UPDATE` or `rewrite_files` now reads on one thread
where it previously used the hand-rolled split. `merge_adjacent_files` is unaffected — it keeps
file-level parallelism, one source exec per file.
