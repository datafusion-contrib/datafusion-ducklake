#![cfg(feature = "metadata-duckdb")]
//! SQL Logic Test Runner using Hybrid DuckDB+DataFusion Adapter
//!
//! This runner executes DuckDB DuckLake tests using a hybrid approach:
//! - WRITE operations (CREATE/INSERT/UPDATE/DELETE) → DuckDB
//! - READ operations (SELECT) → DataFusion
//! - After each WRITE → Refresh DataFusion catalog snapshot
//! - Table references rewritten for DataFusion (ducklake.table → ducklake.main.table)
//!
//! This allows comprehensive testing of DataFusion's read path.
//! Tests from: https://github.com/duckdb/ducklake/tree/main/test/sql

mod common;
mod hybrid_asyncdb;

use hybrid_asyncdb::HybridDuckLakeDB;
use sqllogictest::Runner;
use tempfile::TempDir;

/// Preprocess DuckDB test file to remove DuckDB-specific directives
///
/// This preprocessing:
/// 1. Removes DuckDB-specific test directives (require, test-env, etc.)
/// 2. Skips ATTACH/DETACH statements (handled in Rust)
/// 3. Skips EXPLAIN statements (not testable in hybrid mode)
/// 4. Otherwise lets tests fail naturally for better diagnostics
fn preprocess_test_file(content: &str) -> String {
    let mut output = String::new();
    let mut lines = content.lines().peekable();
    // Arity of the most recent `query <types>` directive, used to expand
    // DuckDB's `ORDER BY ALL` (which DataFusion lacks) into `ORDER BY 1, ...`.
    let mut last_query_arity: usize = 0;

    while let Some(line) = lines.next() {
        let trimmed = line.trim();

        if let Some(types) = trimmed.strip_prefix("query ") {
            let types = types.split_whitespace().next().unwrap_or("");
            if !types.is_empty() && types.chars().all(|c| "TIRB".contains(c)) {
                last_query_arity = types.len();
            }
        }
        let line = if last_query_arity > 0 && line.to_uppercase().contains("ORDER BY ALL") {
            let cols: Vec<String> = (1..=last_query_arity).map(|i| i.to_string()).collect();
            let replacement = format!("ORDER BY {}", cols.join(", "));
            let start = line.to_uppercase().find("ORDER BY ALL").unwrap();
            format!(
                "{}{}{}",
                &line[..start],
                replacement,
                &line[start + "ORDER BY ALL".len()..]
            )
        } else {
            line.to_string()
        };
        let line = line.as_str();
        let trimmed = line.trim();

        // Skip only DuckDB-specific directives that sqllogictest can't parse
        if trimmed.starts_with("require ")
            || trimmed.starts_with("test-env ")
            || trimmed.starts_with("# name:")
            || trimmed.starts_with("# description:")
            || trimmed.starts_with("# group:")
        {
            continue;
        }

        // Skip ATTACH/DETACH statements (we handle connection in Rust)
        if trimmed == "statement ok"
            && let Some(next_line) = lines.peek()
        {
            let next_upper = next_line.trim().to_uppercase();
            if next_upper.starts_with("ATTACH ") || next_upper.starts_with("DETACH ") {
                lines.next(); // Skip the ATTACH/DETACH statement
                continue;
            }

            // Skip EXPLAIN statements (not testable - no consistent output format)
            if next_upper.starts_with("EXPLAIN ") {
                lines.next(); // Skip the EXPLAIN statement
                continue;
            }
        }

        // Skip query blocks with EXPLAIN
        if trimmed.starts_with("query")
            && let Some(next_line) = lines.peek()
        {
            let next_upper = next_line.trim().to_uppercase();
            if next_upper.starts_with("EXPLAIN ") {
                // Skip the query directive, SQL, separator, and results
                lines.next(); // Skip SQL line
                skip_query_results(&mut lines);
                continue;
            }
        }

        // DuckDB's FROM-first shorthand (`FROM t ...`) plans with zero
        // projections in DataFusion; rewrite it to `SELECT * FROM`. Only the
        // FIRST line of a query's SQL qualifies — a continuation line starting
        // with FROM belongs to an ordinary SELECT.
        let starts_sql = output
            .lines()
            .last()
            .map(|prev| {
                let p = prev.trim();
                p.starts_with("query") || p.starts_with("statement")
            })
            .unwrap_or(false);
        if starts_sql && trimmed.to_uppercase().starts_with("FROM ") {
            let indent_len = line.len() - line.trim_start().len();
            output.push_str(&line[..indent_len]);
            output.push_str("SELECT * ");
            output.push_str(line.trim_start());
            output.push('\n');
            continue;
        }

        // Add line as-is (let tests fail naturally on unsupported features)
        output.push_str(line);
        output.push('\n');
    }

    output
}

/// Skip query results until next directive
fn skip_query_results(lines: &mut std::iter::Peekable<std::str::Lines>) {
    // Skip until we find the separator (----)
    while let Some(line) = lines.peek() {
        if line.trim() == "----" {
            lines.next();
            break;
        }
        lines.next();
    }

    // Skip result lines until next directive
    while let Some(line) = lines.peek() {
        let trimmed = line.trim();
        if trimmed.starts_with("query")
            || trimmed.starts_with("statement")
            || trimmed.starts_with("halt")
            || trimmed.is_empty()
        {
            break;
        }
        lines.next();
    }
}

/// Run a DuckDB test file using the hybrid adapter
async fn run_hybrid_test(test_file: &str) -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let catalog_path = temp_dir.path().join("test.ducklake");

    // Create hybrid DB adapter wrapped in Arc for cloning
    let db = std::sync::Arc::new(std::sync::Mutex::new(HybridDuckLakeDB::new(catalog_path)?));

    // Read and preprocess test file
    let original_content = std::fs::read_to_string(test_file)?;
    let processed_content = preprocess_test_file(&original_content);

    // Write preprocessed test to temp file
    let temp_test_file = temp_dir.path().join("test.slt");
    std::fs::write(&temp_test_file, processed_content)?;

    // Run preprocessed test file with sqllogictest
    let temp_test_path = temp_test_file.to_string_lossy().to_string();
    tokio::task::spawn_blocking(move || {
        let mut runner = Runner::new(|| {
            let db_clone = db.clone();
            async move { Ok(db_clone.lock().unwrap().clone()) }
        });
        runner.run_file(&temp_test_path)
    })
    .await??;

    Ok(())
}

// ============================================================================
// Auto-discovery test runner - runs all .test files
// ============================================================================

/// Vendored-suite files that MUST pass: the build fails if any regresses.
/// Newly passing files are reported so this list can grow (a ratchet).
///
/// The rest of the vendored suite is informational — most files exercise
/// DuckDB-specific directives (`loop`, `foreach`, `SET VARIABLE`), types, or
/// virtual columns (`file_row_number`) the hybrid runner does not translate.
/// CDC conformance is authoritatively covered by tests/cdc_differential_tests.rs,
/// which diffs live against the official extension.
const EXPECTED_PASS: &[&str] = &[
    "attach/attach_replace.test",
    "catalog/quoted_identifiers.test",
    // data_inlining_alter.test is EXCLUDED: it asserts unordered `SELECT *`
    // output across two data files, whose row order varies with DataFusion's
    // scan parallelism (machine-dependent).
    "data_inlining/data_inlining_issue504.test",
    "data_inlining/data_inlining_table_changes.test",
    "insert/insert_column_list.test",
    "partitioning/multi_table_partition.test",
    "stats/cardinality.test",
    "stats/filter_pushdown.test",
];

#[tokio::test]
async fn run_all_sqllogictests() {
    use std::path::Path;

    let test_dir = Path::new("tests/sqllogictests/sql");
    let mut test_files = Vec::new();

    // Recursively find all .test files
    fn find_test_files(dir: &Path, files: &mut Vec<std::path::PathBuf>) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    find_test_files(&path, files);
                } else if path.extension().is_some_and(|ext| ext == "test") {
                    files.push(path);
                }
            }
        }
    }

    find_test_files(test_dir, &mut test_files);
    test_files.sort();

    println!("\nFound {} test files", test_files.len());

    let mut passed = 0;
    let mut failed = 0;
    let mut failed_tests = Vec::new();
    let mut passed_tests: Vec<String> = Vec::new();

    for test_file in &test_files {
        let test_name = test_file
            .strip_prefix("tests/sqllogictests/sql/")
            .unwrap_or(test_file.as_path())
            .display()
            .to_string();

        match run_hybrid_test(test_file.to_str().unwrap()).await {
            Ok(_) => {
                println!("✓ {}", test_name);
                passed += 1;
                passed_tests.push(test_name.clone());
            },
            Err(e) => {
                println!("✗ {}: {}", test_name, e);
                failed += 1;
                failed_tests.push((test_name, e.to_string()));
            },
        }
    }

    println!("\n========================================");
    println!("Test Summary:");
    println!("  Passed: {}", passed);
    println!("  Failed: {}", failed);
    println!("  Total:  {}", test_files.len());
    println!("========================================");

    if !failed_tests.is_empty() {
        println!("\nFailed tests:");
        for (name, error) in &failed_tests {
            println!("  - {}", name);
            // Print first line of error only
            if let Some(first_line) = error.lines().next() {
                println!("    {}", first_line);
            }
        }
    }

    // The ratchet: every EXPECTED_PASS file must pass; newly passing files are
    // surfaced so the list can grow.
    let regressions: Vec<&(String, String)> = failed_tests
        .iter()
        .filter(|(name, _)| EXPECTED_PASS.contains(&name.as_str()))
        .collect();
    let newly_passing: Vec<&String> = passed_tests
        .iter()
        .filter(|name| !EXPECTED_PASS.contains(&name.as_str()))
        .collect();
    if !newly_passing.is_empty() {
        println!(
            "\nNewly passing (add to EXPECTED_PASS): {:?}",
            newly_passing
        );
    }
    assert!(
        regressions.is_empty(),
        "sqllogictest regressions in EXPECTED_PASS files: {:?}",
        regressions
            .iter()
            .map(|(name, err)| format!("{name}: {}", err.lines().next().unwrap_or("")))
            .collect::<Vec<_>>()
    );
}
