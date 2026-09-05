//! Integration tests for renamed column support
//!
//! Tests the fix for GitHub issue #24: Renamed columns produce incorrect query results
//! https://github.com/hotdata-dev/datafusion-ducklake/issues/24
//!
//! When columns are renamed in DuckLake, the Parquet files retain original column names
//! but with field_id metadata. DuckLake metadata stores column_id = Parquet field_id.
//! These tests verify that queries work correctly after column renames.

#![cfg(feature = "metadata-duckdb")]

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use arrow::array::{Array, Int32Array, StringArray};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use datafusion_ducklake::{DuckLakeCatalog, DuckdbMetadataProvider};
use tempfile::TempDir;

/// Creates a catalog with a renamed column
///
/// Table schema:
/// - test_table (new_id INT, name VARCHAR)  -- originally (id INT, name VARCHAR)
/// - 3 rows: (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')
///
/// The Parquet file has columns: (id, name) with field_ids (1, 2)
/// The metadata has columns: (new_id, name) with column_ids (1, 2)
fn create_catalog_with_renamed_column(catalog_path: &Path) -> Result<()> {
    let conn = duckdb::Connection::open_in_memory()?;

    crate::common::ensure_ducklake_installed();
    conn.execute("LOAD ducklake;", [])?;

    crate::common::attach_catalog_without_inlining(&conn, catalog_path, "test_catalog")?;

    // Create table with original column name
    conn.execute(
        "CREATE TABLE test_catalog.test_table (
            id INT,
            name VARCHAR
        );",
        [],
    )?;

    // Insert data (creates Parquet file with original column names)
    conn.execute(
        "INSERT INTO test_catalog.test_table VALUES
            (1, 'Alice'),
            (2, 'Bob'),
            (3, 'Charlie');",
        [],
    )?;

    // Rename the column (updates metadata but not Parquet file)
    conn.execute(
        "ALTER TABLE test_catalog.test_table RENAME COLUMN id TO new_id;",
        [],
    )?;

    Ok(())
}

/// Creates a catalog with multiple renamed columns
fn create_catalog_with_multiple_renames(catalog_path: &Path) -> Result<()> {
    let conn = duckdb::Connection::open_in_memory()?;

    crate::common::ensure_ducklake_installed();
    conn.execute("LOAD ducklake;", [])?;

    crate::common::attach_catalog_without_inlining(&conn, catalog_path, "test_catalog")?;

    // Create table with original column names
    conn.execute(
        "CREATE TABLE test_catalog.multi_rename (
            user_id INT,
            first_name VARCHAR,
            last_name VARCHAR
        );",
        [],
    )?;

    // Insert data
    conn.execute(
        "INSERT INTO test_catalog.multi_rename VALUES
            (1, 'John', 'Doe'),
            (2, 'Jane', 'Smith');",
        [],
    )?;

    // Rename multiple columns
    conn.execute(
        "ALTER TABLE test_catalog.multi_rename RENAME COLUMN user_id TO userId;",
        [],
    )?;
    conn.execute(
        "ALTER TABLE test_catalog.multi_rename RENAME COLUMN first_name TO firstName;",
        [],
    )?;
    conn.execute(
        "ALTER TABLE test_catalog.multi_rename RENAME COLUMN last_name TO lastName;",
        [],
    )?;

    Ok(())
}

/// Helper to get int column values from a batch
fn get_int_column(batch: &RecordBatch, col_idx: usize) -> Vec<i32> {
    let column = batch.column(col_idx);
    let array = column.as_any().downcast_ref::<Int32Array>().unwrap();
    array.values().to_vec()
}

/// Helper to get string column values from a batch
fn get_string_column(batch: &RecordBatch, col_idx: usize) -> Vec<String> {
    // Cast to Utf8 so the helper reads any string layout the provider produces
    // (Utf8 / LargeUtf8 / Utf8View); DuckLake string columns now scan as Utf8View.
    let column =
        arrow::compute::cast(batch.column(col_idx), &arrow::datatypes::DataType::Utf8).unwrap();
    let array = column.as_any().downcast_ref::<StringArray>().unwrap();
    (0..array.len())
        .map(|i| array.value(i).to_string())
        .collect()
}

#[tokio::test]
async fn test_select_all_after_rename() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let catalog_path = temp_dir.path().join("renamed.ducklake");

    create_catalog_with_renamed_column(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let catalog = DuckLakeCatalog::new(provider)?;

    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    // Query should use renamed column name
    let df = ctx
        .sql("SELECT new_id, name FROM ducklake.main.test_table ORDER BY new_id")
        .await?;

    let batches = df.collect().await?;
    assert_eq!(batches.len(), 1);

    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 3);

    // Verify data is correct
    let ids = get_int_column(batch, 0);
    assert_eq!(ids, vec![1, 2, 3]);

    let names = get_string_column(batch, 1);
    assert_eq!(names, vec!["Alice", "Bob", "Charlie"]);

    Ok(())
}

#[tokio::test]
async fn test_select_renamed_column_only() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let catalog_path = temp_dir.path().join("renamed.ducklake");

    create_catalog_with_renamed_column(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let catalog = DuckLakeCatalog::new(provider)?;

    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    // Select only the renamed column
    let df = ctx
        .sql("SELECT new_id FROM ducklake.main.test_table ORDER BY new_id")
        .await?;

    let batches = df.collect().await?;
    let batch = &batches[0];

    let ids = get_int_column(batch, 0);
    assert_eq!(ids, vec![1, 2, 3]);

    Ok(())
}

#[tokio::test]
async fn test_filter_on_renamed_column() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let catalog_path = temp_dir.path().join("renamed.ducklake");

    create_catalog_with_renamed_column(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let catalog = DuckLakeCatalog::new(provider)?;

    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    // Filter using the renamed column name
    let df = ctx
        .sql("SELECT new_id, name FROM ducklake.main.test_table WHERE new_id > 1 ORDER BY new_id")
        .await?;

    let batches = df.collect().await?;
    let batch = &batches[0];

    assert_eq!(batch.num_rows(), 2);

    let ids = get_int_column(batch, 0);
    assert_eq!(ids, vec![2, 3]);

    let names = get_string_column(batch, 1);
    assert_eq!(names, vec!["Bob", "Charlie"]);

    Ok(())
}

#[tokio::test]
async fn test_aggregation_on_renamed_column() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let catalog_path = temp_dir.path().join("renamed.ducklake");

    create_catalog_with_renamed_column(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let catalog = DuckLakeCatalog::new(provider)?;

    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    // Aggregate on renamed column
    let df = ctx
        .sql("SELECT SUM(new_id) as total FROM ducklake.main.test_table")
        .await?;

    let batches = df.collect().await?;
    let batch = &batches[0];

    // Sum of 1+2+3 = 6
    let total = batch
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(total, 6);

    Ok(())
}

#[tokio::test]
async fn test_multiple_renamed_columns() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let catalog_path = temp_dir.path().join("multi_rename.ducklake");

    create_catalog_with_multiple_renames(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let catalog = DuckLakeCatalog::new(provider)?;

    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    // Query with all renamed column names (quoted for case sensitivity)
    let df = ctx
        .sql("SELECT \"userId\", \"firstName\", \"lastName\" FROM ducklake.main.multi_rename ORDER BY \"userId\"")
        .await?;

    let batches = df.collect().await?;
    let batch = &batches[0];

    assert_eq!(batch.num_rows(), 2);

    let ids = get_int_column(batch, 0);
    assert_eq!(ids, vec![1, 2]);

    let first_names = get_string_column(batch, 1);
    assert_eq!(first_names, vec!["John", "Jane"]);

    let last_names = get_string_column(batch, 2);
    assert_eq!(last_names, vec!["Doe", "Smith"]);

    Ok(())
}

#[tokio::test]
async fn test_count_after_rename() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let catalog_path = temp_dir.path().join("renamed.ducklake");

    create_catalog_with_renamed_column(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let catalog = DuckLakeCatalog::new(provider)?;

    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    let df = ctx
        .sql("SELECT COUNT(*) as cnt FROM ducklake.main.test_table")
        .await?;

    let batches = df.collect().await?;
    let batch = &batches[0];

    let count = batch
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 3);

    Ok(())
}

#[tokio::test]
async fn test_schema_shows_renamed_columns() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let catalog_path = temp_dir.path().join("renamed.ducklake");

    create_catalog_with_renamed_column(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let catalog = DuckLakeCatalog::new(provider)?;

    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    let df = ctx
        .sql("SELECT * FROM ducklake.main.test_table LIMIT 1")
        .await?;

    // Check schema has renamed column
    let schema = df.schema();
    let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();

    assert!(
        field_names.contains(&"new_id"),
        "Schema should contain 'new_id' not 'id'"
    );
    assert!(field_names.contains(&"name"));

    Ok(())
}

/// Repro: a column is renamed AFTER the first file is written, then a SECOND
/// file is written post-rename. The two parquet files carry the same field_id
/// under DIFFERENT physical names (`id` vs `new_id`). The read path must map
/// field_id -> physical name PER FILE; deriving one mapping from the first file
/// and applying it to every file reads the other file's renamed column as NULL.
fn create_catalog_renamed_with_post_rename_file(catalog_path: &Path) -> Result<()> {
    let conn = duckdb::Connection::open_in_memory()?;
    crate::common::ensure_ducklake_installed();
    conn.execute("LOAD ducklake;", [])?;
    crate::common::attach_catalog_without_inlining(&conn, catalog_path, "test_catalog")?;
    conn.execute(
        "CREATE TABLE test_catalog.test_table (id INT, name VARCHAR);",
        [],
    )?;
    // File 1: physical column `id`.
    conn.execute(
        "INSERT INTO test_catalog.test_table VALUES (1,'Alice'),(2,'Bob'),(3,'Charlie');",
        [],
    )?;
    conn.execute(
        "ALTER TABLE test_catalog.test_table RENAME COLUMN id TO new_id;",
        [],
    )?;
    // File 2: written AFTER the rename -> physical column `new_id`.
    conn.execute(
        "INSERT INTO test_catalog.test_table VALUES (4,'Dave'),(5,'Eve');",
        [],
    )?;
    Ok(())
}

#[tokio::test]
async fn test_rename_with_post_rename_file_reads_all_rows() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let catalog_path = temp_dir.path().join("renamed_multifile.ducklake");
    create_catalog_renamed_with_post_rename_file(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    let df = ctx
        .sql("SELECT new_id FROM ducklake.main.test_table")
        .await?;
    let batches = df.collect().await?;

    let mut ids: Vec<i32> = Vec::new();
    let mut nulls = 0usize;
    for b in &batches {
        let a = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..a.len() {
            if a.is_null(i) {
                nulls += 1;
            } else {
                ids.push(a.value(i));
            }
        }
    }
    ids.sort();
    assert_eq!(
        nulls, 0,
        "renamed column read NULL (post-rename file mis-mapped)"
    );
    assert_eq!(
        ids,
        vec![1, 2, 3, 4, 5],
        "all rows from both files read correctly"
    );
    Ok(())
}

// ===== Schema-evolution regression tests =====

/// SELECT * (projection=None) over rename + post-rename file.
#[tokio::test]
async fn test_select_star_rename_multifile_reads_all_rows() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let catalog_path = temp_dir.path().join("star.ducklake");
    create_catalog_renamed_with_post_rename_file(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    let df = ctx
        .sql("SELECT * FROM ducklake.main.test_table ORDER BY new_id")
        .await?;
    let batches = df.collect().await?;
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 5, "select * row count");
    // verify new_id has no nulls
    let mut nulls = 0;
    for b in &batches {
        let a = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..a.len() {
            if a.is_null(i) {
                nulls += 1;
            }
        }
    }
    assert_eq!(nulls, 0, "select * new_id nulls");
    Ok(())
}

/// Rename a column, write a post-rename file, then DELETE a row from the
/// pre-rename file. This mixes the no-delete group (post-rename file) with the
/// with-delete path (pre-rename file). Both must union to a consistent schema
/// and read the renamed column correctly.
fn create_catalog_rename_then_delete(catalog_path: &Path) -> Result<()> {
    let conn = duckdb::Connection::open_in_memory()?;
    crate::common::ensure_ducklake_installed();
    conn.execute("LOAD ducklake;", [])?;
    crate::common::attach_catalog_without_inlining(&conn, catalog_path, "test_catalog")?;
    conn.execute(
        "CREATE TABLE test_catalog.test_table (id INT, name VARCHAR);",
        [],
    )?;
    conn.execute(
        "INSERT INTO test_catalog.test_table VALUES (1,'Alice'),(2,'Bob'),(3,'Charlie');",
        [],
    )?;
    conn.execute(
        "ALTER TABLE test_catalog.test_table RENAME COLUMN id TO new_id;",
        [],
    )?;
    conn.execute(
        "INSERT INTO test_catalog.test_table VALUES (4,'Dave'),(5,'Eve');",
        [],
    )?;
    // delete a row from the FIRST (pre-rename) file -> produces a delete file
    conn.execute("DELETE FROM test_catalog.test_table WHERE new_id = 2;", [])?;
    Ok(())
}

#[tokio::test]
async fn test_rename_mixed_with_delete_reads_all_rows() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let catalog_path = temp_dir.path().join("mixdel.ducklake");
    create_catalog_rename_then_delete(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    let df = ctx
        .sql("SELECT new_id FROM ducklake.main.test_table")
        .await?;
    let batches = df.collect().await?;
    let mut ids: Vec<i32> = Vec::new();
    let mut nulls = 0usize;
    for b in &batches {
        let a = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..a.len() {
            if a.is_null(i) {
                nulls += 1;
            } else {
                ids.push(a.value(i));
            }
        }
    }
    ids.sort();
    assert_eq!(nulls, 0, "mixed rename+delete: renamed col read NULL");
    assert_eq!(
        ids,
        vec![1, 3, 4, 5],
        "mixed rename+delete: row 2 deleted, rest present"
    );
    Ok(())
}

/// DROP a column then re-ADD a column with the same name. DuckLake assigns a
/// NEW field_id to the re-added column. Old files have the original field_id,
/// new files have the new field_id. Reading the re-added column must return
/// NULL for old files (field_id absent) and values for new files.
fn create_catalog_drop_readd(catalog_path: &Path) -> Result<()> {
    let conn = duckdb::Connection::open_in_memory()?;
    crate::common::ensure_ducklake_installed();
    conn.execute("LOAD ducklake;", [])?;
    crate::common::attach_catalog_without_inlining(&conn, catalog_path, "test_catalog")?;
    conn.execute(
        "CREATE TABLE test_catalog.test_table (id INT, tag VARCHAR);",
        [],
    )?;
    conn.execute(
        "INSERT INTO test_catalog.test_table VALUES (1,'x'),(2,'y');",
        [],
    )?;
    conn.execute("ALTER TABLE test_catalog.test_table DROP COLUMN tag;", [])?;
    conn.execute(
        "ALTER TABLE test_catalog.test_table ADD COLUMN tag VARCHAR;",
        [],
    )?;
    conn.execute("INSERT INTO test_catalog.test_table VALUES (3,'z');", [])?;
    Ok(())
}

#[tokio::test]
async fn test_drop_readd_column_reads_null_for_pre_drop_rows() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let catalog_path = temp_dir.path().join("dropreadd.ducklake");
    create_catalog_drop_readd(&catalog_path)?;

    // The re-added `tag` must read NULL for rows written before the DROP and 'z'
    // for the row written after the re-ADD — NOT the dropped column's stale data.
    let expected: Vec<(i32, Option<String>)> =
        vec![(1, None), (2, None), (3, Some("z".to_string()))];

    // Pin the expectation to DuckDB's own answer so it stays honest.
    {
        let conn = duckdb::Connection::open_in_memory()?;
        crate::common::ensure_ducklake_installed();
        conn.execute("LOAD ducklake;", [])?;
        crate::common::attach_catalog_without_inlining(&conn, &catalog_path, "test_catalog")?;
        let mut stmt = conn.prepare("SELECT id, tag FROM test_catalog.test_table ORDER BY id")?;
        let mut rows = stmt.query([])?;
        let mut duck: Vec<(i32, Option<String>)> = Vec::new();
        while let Some(r) = rows.next()? {
            duck.push((r.get(0)?, r.get(1)?));
        }
        assert_eq!(duck, expected, "DuckDB ground truth changed");
    }

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    let df = ctx
        .sql("SELECT id, tag FROM ducklake.main.test_table ORDER BY id")
        .await?;
    let batches = df.collect().await?;
    let mut got: Vec<(i32, Option<String>)> = Vec::new();
    for b in &batches {
        let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let tags_col =
            arrow::compute::cast(b.column(1), &arrow::datatypes::DataType::Utf8).unwrap();
        let tags = tags_col.as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..b.num_rows() {
            let tag = if tags.is_null(i) {
                None
            } else {
                Some(tags.value(i).to_string())
            };
            got.push((ids.value(i), tag));
        }
    }

    assert_eq!(
        got, expected,
        "re-added column must read NULL for pre-drop rows, not the dropped column's stale data"
    );
    Ok(())
}

/// A struct whose child was touched by DDL: DuckLake records the child as
/// NON-nullable in the catalog while the parquet node stays optional, and files
/// written before the DDL do not carry the child at all.
///
/// Both `ADD COLUMN s.<child>` and `RENAME COLUMN s.<child>` rewrite the child's
/// catalog row this way.
fn create_catalog_evolved_struct_child(catalog_path: &Path, rename: bool) -> Result<()> {
    let conn = duckdb::Connection::open_in_memory()?;
    crate::common::ensure_ducklake_installed();
    conn.execute("LOAD ducklake;", [])?;
    crate::common::attach_catalog_without_inlining(&conn, catalog_path, "test_catalog")?;
    if rename {
        conn.execute(
            "CREATE TABLE test_catalog.t (id INT, s STRUCT(a INT, b INT));",
            [],
        )?;
        conn.execute(
            "INSERT INTO test_catalog.t VALUES (1, {'a': 10, 'b': 20});",
            [],
        )?;
        conn.execute("ALTER TABLE test_catalog.t RENAME COLUMN s.b TO bb;", [])?;
        conn.execute(
            "INSERT INTO test_catalog.t VALUES (2, {'a': 30, 'bb': 40});",
            [],
        )?;
    } else {
        conn.execute("CREATE TABLE test_catalog.t (id INT, s STRUCT(a INT));", [])?;
        conn.execute("INSERT INTO test_catalog.t VALUES (1, {'a': 10});", [])?;
        conn.execute("ALTER TABLE test_catalog.t ADD COLUMN s.b INT;", [])?;
        conn.execute(
            "INSERT INTO test_catalog.t VALUES (2, {'a': 30, 'b': 40});",
            [],
        )?;
    }
    Ok(())
}

async fn evolved_struct_context(catalog_path: &Path) -> Result<SessionContext> {
    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    Ok(ctx)
}

/// The child's catalog nullability must not be enforced against the file: an
/// ordinary `SELECT` over a table whose struct child was added or renamed by DDL
/// has to read the child as NULL where the file predates it, not fail the scan.
#[tokio::test]
async fn test_scan_reads_struct_child_touched_by_ddl() -> Result<()> {
    for (rename, child, expected) in [
        (
            false,
            "b",
            vec![(1, Some(10), None), (2, Some(30), Some(40))],
        ),
        (
            true,
            "bb",
            vec![(1, Some(10), Some(20)), (2, Some(30), Some(40))],
        ),
    ] {
        let temp_dir = TempDir::new()?;
        let catalog_path = temp_dir.path().join("evolved_struct.ducklake");
        create_catalog_evolved_struct_child(&catalog_path, rename)?;
        let ctx = evolved_struct_context(&catalog_path).await?;
        let batches = ctx
            .sql(&format!(
                "SELECT id, s['a'], s['{child}'] FROM ducklake.main.t ORDER BY id"
            ))
            .await?
            .collect()
            .await?;
        let mut got: Vec<(i32, Option<i32>, Option<i32>)> = Vec::new();
        for b in &batches {
            let cell = |col: usize, row: usize| {
                let a = b.column(col).as_any().downcast_ref::<Int32Array>().unwrap();
                (!a.is_null(row)).then(|| a.value(row))
            };
            for row in 0..b.num_rows() {
                got.push((
                    b.column(0)
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .unwrap()
                        .value(row),
                    cell(1, row),
                    cell(2, row),
                ));
            }
        }
        assert_eq!(got, expected, "rename={rename}");
    }
    Ok(())
}

/// A read schema that relaxes nested nullability the way the catalog schema does
/// is type-identical to it, so the rename layer above the scan is a relabel and
/// filters still reach the parquet reader's pruning.
#[tokio::test]
async fn test_filter_pushes_down_over_evolved_struct_column() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let catalog_path = temp_dir.path().join("evolved_struct_pushdown.ducklake");
    create_catalog_evolved_struct_child(&catalog_path, true)?;
    let ctx = evolved_struct_context(&catalog_path).await?;

    // The struct column has to be IN the projection: with it projected away the
    // rename layer only ever sees scalar fields, and pushdown never depended on
    // the nested types at all.
    let plan = ctx
        .sql("EXPLAIN SELECT id, s FROM ducklake.main.t WHERE id = 2")
        .await?
        .collect()
        .await?;
    let rendered = arrow::util::pretty::pretty_format_batches(&plan)?.to_string();
    assert!(
        rendered.contains("predicate=id@0 = 2"),
        "the filter must reach the scan, plan was:\n{rendered}"
    );

    let rows = ctx
        .sql("SELECT id, s FROM ducklake.main.t WHERE id = 2")
        .await?
        .collect()
        .await?;
    let total: usize = rows.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total, 1,
        "the pushed-down filter must keep the matching row"
    );
    Ok(())
}
