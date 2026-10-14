#![cfg(feature = "metadata-duckdb")]
//! SQL Logic Test Runner using Hybrid DuckDB+DataFusion Adapter
//!
//! Runs the official DuckLake sqllogictests vendored under `tests/sqllogictests/sql`
//! (from <https://github.com/duckdb/ducklake/tree/main/test/sql>). Writes and
//! DuckDB-only statements run on DuckDB with the official extension; queries over
//! DuckLake tables run on DataFusion through this crate. See `hybrid_asyncdb.rs`
//! for the routing rules.
//!
//! The runner translates DuckDB's sqllogictest dialect into what the
//! `sqllogictest` crate parses (see [`preprocess_test_file`]):
//!
//! - `require <extension>` installs and loads the extension; `require-env` skips
//!   the file when the variable is unset; platform requirements are dropped.
//! - `test-env NAME VALUE` defines `${NAME}` (also spelled `{NAME}`); `__TEST_DIR__` / `{TEST_DIR}` become
//!   the file's temporary directory and `{UUID}` a unique name.
//! - `loop` / `foreach` bodies are expanded. `concurrentloop` / `concurrentforeach`
//!   run their iterations sequentially: the adapter has one DuckDB connection, so
//!   these files test the end state, not the concurrency.
//! - `mode skip` ... `mode unskip` regions are dropped.
//! - `statement error` with a `----` message becomes an inline regex matching that
//!   message as a substring, as DuckDB's runner matches it.
//! - `statement maybe` accepts success or failure with the given message.
//! - A labelled query with no expected rows (`query I nosort label`) must return
//!   the rows of the first query with that label, as in DuckDB's runner.
//! - Result values compare as DuckDB's runner compares them (see
//!   [`duckdb_validator`]).
//! - `EXPLAIN` records are dropped; `ORDER BY ALL` is spelled out for DataFusion.
//! - A query without `ORDER BY` compares its rows as a sorted set (`rowsort`).
//!
//! A file only counts as passing if DataFusion ran at least one query over a
//! DuckLake table: a file whose reads all landed on DuckDB tested nothing here.
//!
//! `.test_slow` files run only in the ignored `run_slow_sqllogictests`.

use crate::hybrid_asyncdb;

use hybrid_asyncdb::{HybridDuckLakeDB, LABEL_MARKER, LABEL_MATCH, MAYBE_MARKER, RouteStats};
use sqllogictest::Runner;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

/// A test file translated for the `sqllogictest` crate.
struct Preprocessed {
    script: String,
    /// Extensions named by `require`, to install and load before running.
    extensions: Vec<String>,
}

/// Requirements the runner satisfies by construction or that do not apply.
const IGNORED_REQUIREMENTS: &[&str] = &[
    "ducklake",
    "parquet",
    "notwindows",
    "notmingw",
    "64bit",
    "skip_reload",
    "no_alternative_verify",
    "no_extension_autoloading",
];

/// Translate a DuckDB sqllogictest file; `Err` is the reason to skip it.
fn preprocess_test_file(content: &str, test_dir: &Path) -> Result<Preprocessed, String> {
    let test_dir = test_dir.display().to_string();
    let mut unique = 0usize;
    let mut substitute_paths = |line: &str| -> String {
        let mut line = line
            .replace("__TEST_DIR__", &test_dir)
            .replace("{TEST_DIR}", &test_dir);
        while line.contains("{UUID}") {
            unique += 1;
            line = line.replacen("{UUID}", &format!("hybrid_{unique}"), 1);
        }
        line
    };

    // Header directives and skipped regions.
    let mut extensions = Vec::new();
    let mut vars: Vec<(String, String)> = Vec::new();
    let mut lines: Vec<String> = Vec::new();
    let mut skipping = false;
    for line in content.lines() {
        let trimmed = line.trim();
        let mut words = trimmed.split_whitespace();
        match (words.next(), words.next()) {
            (Some("mode"), Some("skip")) => {
                skipping = true;
                continue;
            },
            (Some("mode"), Some("unskip")) => {
                skipping = false;
                continue;
            },
            _ if skipping => continue,
            (Some("require"), Some(requirement)) => {
                if !IGNORED_REQUIREMENTS.contains(&requirement) {
                    extensions.push(requirement.to_string());
                }
                continue;
            },
            (Some("require-env"), Some(var)) => {
                if std::env::var(var).is_err() {
                    return Err(format!("requires environment variable {var}"));
                }
                continue;
            },
            (Some("test-env"), Some(name)) => {
                let value = trimmed
                    .split_whitespace()
                    .skip(2)
                    .collect::<Vec<_>>()
                    .join(" ");
                vars.push((name.to_string(), substitute_paths(&value)));
                continue;
            },
            _ => {},
        }
        if trimmed.starts_with("# name:")
            || trimmed.starts_with("# description:")
            || trimmed.starts_with("# group:")
        {
            continue;
        }
        lines.push(line.to_string());
    }

    let lines = expand_loops(&lines)?;
    let lines: Vec<String> = lines
        .iter()
        .map(|line| {
            let mut line = substitute_paths(line);
            for (name, value) in &vars {
                line = line
                    .replace(&format!("${{{name}}}"), value)
                    .replace(&format!("{{{name}}}"), value);
            }
            line
        })
        .collect();
    // A placeholder left unsubstituted would name a path relative to the
    // process directory, and DuckDB would create it in the repository.
    let placeholder = regex::Regex::new(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}|\{[A-Z][A-Z0-9_]+\}")
        .expect("placeholder pattern");
    if let Some(found) = lines.iter().find_map(|line| placeholder.find(line)) {
        return Err(format!("unresolved placeholder {}", found.as_str()));
    }

    Ok(Preprocessed {
        script: translate_records(&lines),
        extensions,
    })
}

/// Values DuckDB's runner gives the `foreach` type groups.
fn foreach_values(token: &str) -> Vec<String> {
    let integral = [
        "tinyint",
        "smallint",
        "integer",
        "bigint",
        "hugeint",
        "utinyint",
        "usmallint",
        "uinteger",
        "ubigint",
        "uhugeint",
    ];
    let group: &[&str] = match token {
        "<integral>" => &integral,
        "<signed>" => &integral[..5],
        "<unsigned>" => &integral[5..],
        "<numeric>" => {
            return integral
                .iter()
                .chain(["float", "double"].iter())
                .map(|s| s.to_string())
                .collect();
        },
        _ => return vec![token.to_string()],
    };
    group.iter().map(|s| s.to_string()).collect()
}

/// Expand `loop` / `foreach` (and their concurrent forms, run sequentially).
fn expand_loops(lines: &[String]) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let words: Vec<&str> = lines[i].split_whitespace().collect();
        let (variable, values): (String, Vec<String>) = match words.as_slice() {
            ["loop" | "concurrentloop", var, start, end] => {
                let start: i64 = start
                    .parse()
                    .map_err(|_| format!("bad loop: {}", lines[i]))?;
                let end: i64 = end.parse().map_err(|_| format!("bad loop: {}", lines[i]))?;
                (
                    var.to_string(),
                    (start..end).map(|v| v.to_string()).collect(),
                )
            },
            ["foreach" | "concurrentforeach", var, values @ ..] => (
                var.to_string(),
                values.iter().flat_map(|v| foreach_values(v)).collect(),
            ),
            _ => {
                out.push(lines[i].clone());
                i += 1;
                continue;
            },
        };
        // Find the matching `endloop`.
        let mut depth = 0;
        let mut end = None;
        for (j, line) in lines.iter().enumerate().skip(i + 1) {
            match line.split_whitespace().next() {
                Some("loop" | "concurrentloop" | "foreach" | "concurrentforeach") => depth += 1,
                Some("endloop") if depth == 0 => {
                    end = Some(j);
                    break;
                },
                Some("endloop") => depth -= 1,
                _ => {},
            }
        }
        let end = end.ok_or_else(|| format!("unterminated loop: {}", lines[i]))?;
        let body = &lines[i + 1..end];
        let placeholder = format!("${{{variable}}}");
        for value in values {
            let substituted: Vec<String> = body
                .iter()
                .map(|line| line.replace(&placeholder, &value))
                .collect();
            out.extend(expand_loops(&substituted)?);
        }
        i = end + 1;
    }
    Ok(out)
}

/// Whether `line` starts a sqllogictest record.
fn is_record_start(line: &str) -> bool {
    let first = line.split_whitespace().next().unwrap_or("");
    matches!(first, "statement" | "query" | "halt" | "hash-threshold")
}

/// Translate DuckDB record syntax into the `sqllogictest` crate's.
fn translate_records(lines: &[String]) -> String {
    let mut out: Vec<String> = Vec::new();
    // Arity of the most recent `query <types>` directive, used to expand
    // DuckDB's `ORDER BY ALL` (which DataFusion lacks) into `ORDER BY 1, ...`.
    let mut last_query_arity: usize = 0;
    let mut i = 0;
    while i < lines.len() {
        let line = &lines[i];
        let trimmed = line.trim();

        if let Some(types) = trimmed.strip_prefix("query ") {
            let types = types.split_whitespace().next().unwrap_or("");
            if !types.is_empty() && types.chars().all(|c| "TIRB".contains(c)) {
                last_query_arity = types.len();
            }
        }

        let next_upper = lines
            .get(i + 1)
            .map(|l| l.trim().to_uppercase())
            .unwrap_or_default();
        // EXPLAIN output is engine-specific; drop the record.
        if (trimmed.starts_with("statement") || trimmed.starts_with("query"))
            && next_upper.starts_with("EXPLAIN ")
        {
            i += 1;
            while i < lines.len() && !lines[i].trim().is_empty() && !is_record_start(&lines[i]) {
                i += 1;
            }
            continue;
        }

        let words: Vec<&str> = trimmed.split_whitespace().collect();
        if let ["statement", kind @ ("error" | "maybe")] = words.as_slice() {
            // SQL lines, then an optional `----` and expected message.
            let mut sql = Vec::new();
            i += 1;
            while i < lines.len() && !lines[i].trim().is_empty() && lines[i].trim() != "----" {
                sql.push(lines[i].clone());
                i += 1;
            }
            let mut message = Vec::new();
            if i < lines.len() && lines[i].trim() == "----" {
                i += 1;
                while i < lines.len() && !lines[i].trim().is_empty() {
                    message.push(lines[i].trim().to_string());
                    i += 1;
                }
            }
            let message = message.join("\n");
            if *kind == "error" {
                let pattern = match message.strip_prefix("<REGEX>:") {
                    Some(regex) => regex.to_string(),
                    None => regex::escape(&message).replace('\n', "\\n"),
                };
                out.push(format!("statement error {pattern}").trim_end().to_string());
            } else {
                out.push("statement ok".to_string());
                if let Some(first) = sql.first_mut() {
                    *first = format!("{MAYBE_MARKER}{}*/{first}", message.replace("*/", ""));
                }
            }
            out.extend(sql);
            out.push(String::new());
            if i < lines.len() && lines[i].trim().is_empty() {
                i += 1;
            }
            continue;
        }

        // `query <types> <sort> <label>` with no expected rows: DuckDB checks the
        // rows against the first query with that label.
        if let ["query", _, _, label] = words.as_slice() {
            let mut sql = Vec::new();
            let mut j = i + 1;
            while j < lines.len() && !lines[j].trim().is_empty() && lines[j].trim() != "----" {
                sql.push(expand_order_by_all(&lines[j], last_query_arity));
                j += 1;
            }
            let no_rows = j + 1 >= lines.len() || lines[j + 1].trim().is_empty();
            if j < lines.len() && lines[j].trim() == "----" && no_rows && !sql.is_empty() {
                out.push("query T".to_string());
                sql[0] = format!("{LABEL_MARKER}{label}*/{}", sql[0]);
                out.extend(sql);
                out.push("----".to_string());
                out.push(LABEL_MATCH.to_string());
                i = j + 1;
                continue;
            }
        }

        // A query without ORDER BY has no defined row order. DuckDB returns
        // scans in insertion order and the expected rows follow it; DataFusion
        // need not, so such a query compares its rows as a sorted set.
        if let ["query", types] = words.as_slice() {
            let mut sql = Vec::new();
            let mut j = i + 1;
            while j < lines.len() && !lines[j].trim().is_empty() && lines[j].trim() != "----" {
                sql.push(expand_order_by_all(&lines[j], last_query_arity));
                j += 1;
            }
            let mut results = Vec::new();
            if j < lines.len() && lines[j].trim() == "----" {
                let mut k = j + 1;
                while k < lines.len() && !lines[k].trim().is_empty() {
                    results.push(lines[k].clone());
                    k += 1;
                }
                if results.len() > 1 && !sql.join(" ").to_uppercase().contains("ORDER BY") {
                    results.sort();
                    out.push(format!("query {types} rowsort"));
                    out.extend(sql);
                    out.push("----".to_string());
                    out.extend(results);
                    i = k;
                    continue;
                }
            }
        }

        let line = expand_order_by_all(line, last_query_arity);

        out.push(line);
        i += 1;
    }
    let mut script = out.join("\n");
    script.push('\n');
    script
}

fn expand_order_by_all(line: &str, arity: usize) -> String {
    let upper = line.to_uppercase();
    match upper.find("ORDER BY ALL") {
        Some(start) if arity > 0 => {
            let cols: Vec<String> = (1..=arity).map(|i| i.to_string()).collect();
            format!(
                "{}ORDER BY {}{}",
                &line[..start],
                cols.join(", "),
                &line[start + "ORDER BY ALL".len()..]
            )
        },
        _ => line.to_string(),
    }
}

/// Compare results the way DuckDB's runner does: values are tab-separated (or,
/// when no expected value holds a space, space-separated), each row has as many
/// values as expected, and a value that differs as text still matches when its
/// column is boolean and both sides spell the same boolean (`true`, `True`,
/// `1`), when its column is numeric and both sides are the same number (`100`
/// and `100.0`), or when the expected value is a `<REGEX>:` (or `<!REGEX>:`)
/// pattern. The adapter tags each value with its column's kind
/// ([`hybrid_asyncdb::NUMERIC_TAG`], [`hybrid_asyncdb::BOOLEAN_TAG`]).
fn duckdb_validator(actual: &[Vec<String>], expected: &[String]) -> bool {
    if actual.len() != expected.len() {
        return false;
    }
    actual.iter().zip(expected).all(|(row, line)| {
        let tabbed: Vec<&str> = line.split('\t').collect();
        let values = if tabbed.len() == row.len() {
            tabbed
        } else if row
            .iter()
            .all(|value| !hybrid_asyncdb::untagged(value).contains(char::is_whitespace))
        {
            // Expected values separated by spaces rather than tabs.
            line.split_whitespace().collect()
        } else {
            return false;
        };
        values.len() == row.len()
            && row
                .iter()
                .zip(values)
                .all(|(actual, expected)| value_matches(actual, expected.trim()))
    })
}

/// A query's result has as many columns as its `query` header declares, as
/// DuckDB's runner requires; the adapter reports no column types beyond that.
#[expect(clippy::ptr_arg, reason = "sqllogictest's column validator type takes `&Vec`")]
fn column_count_validator(
    actual: &Vec<sqllogictest::DefaultColumnType>,
    expected: &Vec<sqllogictest::DefaultColumnType>,
) -> bool {
    actual.len() == expected.len()
}

fn value_matches(tagged: &str, expected: &str) -> bool {
    let actual = hybrid_asyncdb::untagged(tagged).trim();
    if actual == expected {
        return true;
    }
    for (prefix, should_match) in [("<REGEX>:", true), ("<!REGEX>:", false)] {
        if let Some(pattern) = expected.strip_prefix(prefix) {
            return regex::Regex::new(&format!("(?s)^(?:{pattern})$"))
                .is_ok_and(|regex| regex.is_match(actual) == should_match);
        }
    }
    if tagged.starts_with(hybrid_asyncdb::BOOLEAN_TAG) {
        let boolean = |value: &str| match value.to_lowercase().as_str() {
            "true" | "t" | "1" => Some(true),
            "false" | "f" | "0" => Some(false),
            _ => None,
        };
        return boolean(actual).is_some() && boolean(actual) == boolean(expected);
    }
    if tagged.starts_with(hybrid_asyncdb::NUMERIC_TAG) {
        if let (Ok(actual), Ok(expected)) = (actual.parse::<i128>(), expected.parse::<i128>()) {
            return actual == expected;
        }
        return match (actual.parse::<f64>(), expected.parse::<f64>()) {
            (Ok(actual), Ok(expected)) => actual == expected,
            _ => false,
        };
    }
    false
}

/// How one test file ended.
enum FileOutcome {
    Passed(RouteStats),
    /// Every record passed, but DataFusion read no DuckLake table: the file
    /// exercised only DuckDB, so it does not count as passing.
    Vacuous(RouteStats),
    Skipped(String),
    Failed(String),
}

/// Run a DuckDB test file using the hybrid adapter
async fn run_hybrid_test(test_file: &Path) -> FileOutcome {
    match run_hybrid_test_inner(test_file).await {
        Ok(outcome) => outcome,
        Err(error) => FileOutcome::Failed(error.to_string()),
    }
}

async fn run_hybrid_test_inner(
    test_file: &Path,
) -> Result<FileOutcome, Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    // Relative paths in a test (`ATTACH 'ducklake:x.db'`) resolve against the
    // process directory, so the runner gives every test file absolute ones.
    let name = test_name(test_file);
    if let Some((_, reason)) = SKIPPED_FILES.iter().find(|(file, _)| *file == name) {
        return Ok(FileOutcome::Skipped(reason.to_string()));
    }
    let original_content = std::fs::read_to_string(test_file)?;
    let processed = match preprocess_test_file(&original_content, temp_dir.path()) {
        Ok(processed) => processed,
        Err(reason) => return Ok(FileOutcome::Skipped(reason)),
    };

    let db = HybridDuckLakeDB::new()?;
    for extension in &processed.extensions {
        if let Err(error) = db.load_extension(extension) {
            return Ok(FileOutcome::Skipped(format!(
                "require {extension}: {error}"
            )));
        }
    }

    let temp_test_file = temp_dir.path().join("test.slt");
    std::fs::write(&temp_test_file, processed.script)?;

    let runner_db = db.clone();
    tokio::task::spawn_blocking(move || {
        let mut runner = Runner::new(|| {
            let db = runner_db.clone();
            async move { Ok(db) }
        });
        runner.with_validator(duckdb_validator);
        runner.with_column_validator(column_count_validator);
        runner.run_file(&temp_test_file)
    })
    .await??;

    let stats = db.stats();
    if stats.datafusion_lake_reads == 0 {
        return Ok(FileOutcome::Vacuous(stats));
    }
    Ok(FileOutcome::Passed(stats))
}

// ============================================================================
// Auto-discovery test runner - runs all .test files
// ============================================================================

/// Vendored-suite files that MUST pass: the build fails if any regresses.
/// Newly passing files are reported so this list can grow (a ratchet).
///
/// The rest of the vendored suite is informational: those files exercise
/// DuckDB features this crate does not implement, or are listed in
/// [`KNOWN_DIVERGENCES`]. CDC conformance is authoritatively covered by
/// tests/it/cdc_differential_tests.rs, which diffs live against the official
/// extension.
const EXPECTED_PASS: &[&str] = &[
    "add_files/add_empty_file.test",
    "add_files/add_file_specific_schema.test",
    "add_files/add_files.test",
    "add_files/add_files_extra_columns.test",
    "add_files/add_files_hive.test",
    "add_files/add_files_list.test",
    "add_files/add_files_missing_columns.test",
    "add_files/add_files_missing_fields.test",
    "add_files/add_files_nested.test",
    "add_files/add_files_rename.test",
    "add_files/add_files_table_changes.test",
    "add_files/add_files_transaction_local.test",
    "add_files/add_files_type_check_decimal.test",
    "add_files/add_files_type_check_float.test",
    "add_files/add_files_type_check_integer.test",
    "add_files/add_files_type_check_nested.test",
    "add_files/add_files_type_check_string_blob.test",
    "add_files/add_files_type_check_timestamp.test",
    "alter/add_column.test",
    "alter/add_column_nested.test",
    "alter/alter_timestamptz_promotion.test",
    "alter/drop_column.test",
    "alter/drop_column_nested.test",
    "alter/mixed_alter.test",
    "alter/mixed_alter2.test",
    "alter/promote_type.test",
    "alter/rename_column.test",
    "alter/rename_table.test",
    "alter/rename_table_case.test",
    "alter/rename_table_dbt_workload.test",
    "audit/test_base_audit.test",
    "catalog/drop_table.test",
    "catalog/schema.test",
    "checkpoint/checkpoint_updates_interleaved.test",
    "checkpoint/many_inserts_transaction.test",
    "cloud/test_cloud_cases.test",
    "compaction/compaction_full_file_delete.test",
    "compaction/compaction_size_limit.test",
    "concurrent/concurrent_insert_conflict.test",
    "concurrent/concurrent_insert_data_inlining.test",
    "concurrent/file_level_conflict.test",
    "data_inlining/basic_data_inlining.test",
    "data_inlining/data_inlining_alter.test",
    "data_inlining/data_inlining_alter_geometry.test",
    "data_inlining/data_inlining_delete.test",
    "data_inlining/data_inlining_filter.test",
    "data_inlining/data_inlining_flush.test",
    "data_inlining/data_inlining_flush_schema.test",
    "data_inlining/data_inlining_flush_sequential_updates.test",
    "data_inlining/data_inlining_interleaved_update.test",
    "data_inlining/data_inlining_issue504.test",
    "data_inlining/data_inlining_option.test",
    "data_inlining/data_inlining_option_transaction_local.test",
    "data_inlining/data_inlining_partitions.test",
    "data_inlining/data_inlining_per_schema_alter.test",
    "data_inlining/data_inlining_table_changes.test",
    "data_inlining/data_inlining_transaction_local_alter.test",
    "data_inlining/data_inlining_transaction_local_alter_type_rename.test",
    "data_inlining/data_inlining_transaction_local_delete.test",
    "data_inlining/data_inlining_txn_delete_visibility.test",
    "data_inlining/data_inlining_types.test",
    "data_inlining/data_inlining_update.test",
    "data_inlining/data_inlining_update_inline_verification.test",
    "data_inlining/empty_inlined_tables_alter_cleanup.test",
    "data_inlining/flush_concurrent_insert.test",
    "data_inlining/inlined_data_flush_cleanup.test",
    "data_inlining/inlined_data_multi_table_fresh_connection.test",
    "data_inlining/inlining_double.test",
    "data_inlining/inlining_enabled_after_create.test",
    "data_inlining/inlining_reserved_column_names.test",
    "data_inlining/insert_inlining_concurrent.test",
    "data_inlining/long_column_name_inlining.test",
    "data_inlining/partition_inlining.test",
    "data_inlining/superseded_inlined_table_flush_drop.test",
    "default/add_column_with_default.test",
    "default/default_values.test",
    "delete/basic_delete.test",
    "delete/delete_join.test",
    "delete/delete_rollback_cleanup.test",
    "delete/delete_same_transaction.test",
    "delete/empty_delete.test",
    "delete/multi_deletes.test",
    "delete/truncate_table.test",
    "deletion_inlining/test_deletion_from_file_and_inserted_inlining.test",
    "deletion_inlining/test_deletion_from_file_and_inserted_inlining_non_sequential.test",
    "deletion_inlining/test_deletion_from_inlined_insertion.test",
    "deletion_inlining/test_deletion_from_inlined_multiple_snapshots.test",
    "deletion_inlining/test_deletion_inlining.test",
    "deletion_inlining/test_deletion_inlining_alter.test",
    "deletion_inlining/test_deletion_inlining_compaction.test",
    "deletion_inlining/test_deletion_inlining_concurrent.test",
    "deletion_inlining/test_deletion_inlining_concurrent_retry.test",
    "deletion_inlining/test_deletion_inlining_large.test",
    "deletion_inlining/test_deletion_inlining_partitions.test",
    "deletion_inlining/test_deletion_inlining_stats.test",
    "deletion_inlining/test_deletion_inlining_table_deletes.test",
    "deletion_inlining/test_deletion_inlining_transaction.test",
    "deletion_inlining/test_deletion_multiple_tables.test",
    "ducklake_basic.test",
    "general/ducklake_read_only.test",
    "general/metadata_cache.test",
    "general/metadata_parameters.test",
    "insert/insert_column_list.test",
    "issues/late_materialization.test",
    "merge/merge_partition.test",
    "merge/merge_timestamp.test",
    "merge/merge_update_insert.test",
    "partitioning/basic_partitioning.test",
    "partitioning/drop_partition_column.test",
    "partitioning/partitioning_alter.test",
    "partitioning/year_month_day.test",
    "rewrite_data_files/insert_delete_loop.test",
    "rowid/ducklake_row_id.test",
    "rowid/ducklake_row_id_update.test",
    "stats/filter_pushdown.test",
    "table_changes/ducklake_table_changes.test",
    "table_changes/ducklake_table_deletions_inlined_flush.test",
    "transaction/create_conflict.test",
    "transaction/transaction_inlining.test",
    "types/floats.test",
    "types/json.test",
    "types/list.test",
    "types/struct.test",
    "update/basic_update.test",
    "update/update_not_null.test",
    "update/update_partitioning.test",
    "update/update_rollback.test",
    "update/update_same_transaction.test",
    "view/ducklake_rename_view.test",
    "view/ducklake_view.test",
    "view/ducklake_view_schema.test",
    "virtualcolumns/ducklake_snapshot_id.test",
];

/// Files that fail for a known divergence between this crate and DuckDB, with
/// the reason. They still run; one that starts passing is reported so it can
/// move to [`EXPECTED_PASS`].
const KNOWN_DIVERGENCES: &[(&str, &str)] = &[
    (
        "data_inlining/data_inlining_encryption.test",
        "DuckDB-encrypted Parquet files are not readable (COMPATIBILITY.md: only PME)",
    ),
    (
        "deletion_inlining/test_deletion_inlining_encryption.test",
        "DuckDB-encrypted Parquet files are not readable (COMPATIBILITY.md: only PME)",
    ),
    (
        "deletion_inlining/test_deletion_inlining_encryption_merge.test",
        "DuckDB-encrypted Parquet files are not readable (COMPATIBILITY.md: only PME)",
    ),
    (
        "data_inlining/data_inlining_variant.test",
        "the VARIANT type is not supported",
    ),
    (
        "catalog/quoted_identifiers.test",
        "metadata tables in a non-default METADATA_SCHEMA are not read; it left EXPECTED_PASS \
         when the runner began reading its tables through DataFusion",
    ),
];

/// Files that run but read no DuckLake table through DataFusion, so they test
/// nothing of this crate and cannot count as passing, with the reason. Their
/// statements still run on DuckDB, and one that starts reading a DuckLake
/// table through DataFusion is reported like any other file.
const VACUOUS_FILES: &[(&str, &str)] = &[
    (
        "attach/attach_replace.test",
        "only ATTACH statements; no query",
    ),
    (
        "partitioning/multi_table_partition.test",
        "only statements; its checks are DuckDB-side statement outcomes",
    ),
    (
        "stats/cardinality.test",
        "its one query is DuckDB's EXPLAIN",
    ),
    (
        "data_inlining/data_inlining_constraints.test",
        "only statements: DuckDB's constraint errors on inlined inserts",
    ),
    (
        "data_inlining/data_inlining_large.test",
        "its queries read a DuckDB table and the data directory (GLOB)",
    ),
    (
        "data_inlining/data_inlining_prepare.test",
        "only prepared-statement inserts; no query",
    ),
    (
        "data_inlining/inlined_data_table_leak.test",
        "its queries read the metadata database (duckdb_tables, the registry)",
    ),
    (
        "data_inlining/inlined_delete_table_drop_cleanup.test",
        "its queries read the metadata database (duckdb_tables, the registry)",
    ),
    (
        "data_inlining/inlining_global_options.test",
        "its queries read metadata tables and DuckDB's flush table function",
    ),
    (
        "data_inlining/inlining_issue_on_empty_inline.test",
        "its queries are DuckDB's flush table function",
    ),
    (
        "data_inlining/inlining_unsupported_systems.test",
        "only statements: the error for metadata systems without inlining",
    ),
    (
        "data_inlining/set_inlining_limit_after_attach.test",
        "its queries read DuckDB's `filename` virtual column of a DuckDB view",
    ),
    (
        "data_inlining/table_stats.test",
        "its queries read DuckDB's flush table function and a metadata table",
    ),
];

/// Files the runner does not run, with the reason.
const SKIPPED_FILES: &[(&str, &str)] = &[(
    "data_inlining/postgres_identifier_limit.test",
    "expects PostgreSQL metadata's 63-character identifier limit; on the DuckDB metadata \
     the runner uses, the official extension inlines these tables too",
)];

fn test_name(test_file: &Path) -> String {
    test_file
        .strip_prefix("tests/sqllogictests/sql/")
        .unwrap_or(test_file)
        .display()
        .to_string()
}

fn find_test_files(dir: &Path, extension: &str, files: &mut Vec<PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                find_test_files(&path, extension, files);
            } else if path.extension().is_some_and(|ext| ext == extension) {
                files.push(path);
            }
        }
    }
}

/// Run every file with `extension` and return (passed, failed, skipped) names.
async fn run_suite(extension: &str) -> (Vec<String>, Vec<(String, String)>, Vec<String>) {
    // Returns (passed, failed, not counted: skipped or vacuous).
    let test_dir = Path::new("tests/sqllogictests/sql");
    let mut test_files = Vec::new();
    find_test_files(test_dir, extension, &mut test_files);
    test_files.sort();
    if let Ok(filter) = std::env::var("SLT_FILTER") {
        test_files.retain(|path| path.to_string_lossy().contains(&filter));
    }

    println!("\nFound {} .{extension} files", test_files.len());

    let mut passed = Vec::new();
    let mut failed = Vec::new();
    let mut skipped = Vec::new();
    // Passing files whose DataFusion side read a DuckLake table only once.
    let mut single_read = Vec::new();
    for test_file in &test_files {
        let test_name = test_name(test_file);
        match run_hybrid_test(test_file).await {
            FileOutcome::Passed(stats) => {
                println!("✓ {test_name} ({})", stats.summary());
                if stats.datafusion_lake_reads == 1 {
                    single_read.push(test_name.clone());
                }
                passed.push(test_name);
            },
            FileOutcome::Vacuous(stats) => {
                let reason = VACUOUS_FILES
                    .iter()
                    .find(|(file, _)| *file == test_name)
                    .map_or(String::new(), |(_, reason)| format!(": {reason}"));
                println!(
                    "○ {test_name}: vacuous, DataFusion read no DuckLake table ({}){reason}",
                    stats.summary()
                );
                skipped.push(test_name);
            },
            FileOutcome::Skipped(reason) => {
                println!("- {test_name}: skipped, {reason}");
                skipped.push(test_name);
            },
            FileOutcome::Failed(error) => {
                println!("✗ {test_name}: {error}");
                failed.push((test_name, error));
            },
        }
    }

    println!("\n========================================");
    println!("Test Summary:");
    println!("  Passed:  {}", passed.len());
    println!("  Failed:  {}", failed.len());
    println!("  Vacuous or skipped: {}", skipped.len());
    println!(
        "  Passed with exactly one DataFusion read: {}",
        single_read.len()
    );
    println!("  Total:   {}", test_files.len());
    println!("========================================");

    if !failed.is_empty() {
        println!("\nFailed tests:");
        for (name, error) in &failed {
            let divergence = KNOWN_DIVERGENCES.iter().find(|(file, _)| file == name);
            match divergence {
                Some((_, reason)) => println!("  - {name} (known divergence: {reason})"),
                None => println!("  - {name}"),
            }
            if let Some(first_line) = error.lines().next() {
                println!("    {}", first_line);
            }
        }
    }
    (passed, failed, skipped)
}

#[tokio::test]
async fn run_all_sqllogictests() {
    let (passed, failed, _) = run_suite("test").await;

    // The ratchet: every EXPECTED_PASS file must pass; newly passing files are
    // surfaced so the list can grow.
    let regressions: Vec<&(String, String)> = failed
        .iter()
        .filter(|(name, _)| EXPECTED_PASS.contains(&name.as_str()))
        .collect();
    let newly_passing: Vec<&String> = passed
        .iter()
        .filter(|name| !EXPECTED_PASS.contains(&name.as_str()))
        .collect();
    if !newly_passing.is_empty() {
        println!(
            "\nNewly passing (add to EXPECTED_PASS): {:?}",
            newly_passing
        );
    }
    if std::env::var("SLT_FILTER").is_err() {
        let missing: Vec<&&str> = EXPECTED_PASS
            .iter()
            .filter(|name| {
                !passed.iter().any(|p| p == *name) && !failed.iter().any(|(f, _)| f == *name)
            })
            .collect();
        assert!(
            missing.is_empty(),
            "EXPECTED_PASS files that did not run: {missing:?}"
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

/// `.test_slow` files, which official DuckLake runs only in its slow CI job.
const EXPECTED_PASS_SLOW: &[&str] = &["deletion_inlining/test_multiple_files_flush.test_slow"];

#[tokio::test]
#[ignore = "slow: official DuckLake's .test_slow files"]
async fn run_slow_sqllogictests() {
    let (_, failed, _) = run_suite("test_slow").await;
    let regressions: Vec<&(String, String)> = failed
        .iter()
        .filter(|(name, _)| EXPECTED_PASS_SLOW.contains(&name.as_str()))
        .collect();
    assert!(
        regressions.is_empty(),
        "slow sqllogictest regressions: {regressions:?}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loop_expansion() {
        let lines: Vec<String> =
            ["loop i 0 2", "SELECT ${i}", "foreach t a b", "${t}${i}", "endloop", "endloop"]
                .iter()
                .map(|s| s.to_string())
                .collect();
        assert_eq!(
            expand_loops(&lines).unwrap(),
            ["SELECT 0", "a0", "b0", "SELECT 1", "a1", "b1"]
        );
    }

    /// A placeholder the runner cannot substitute refuses the file instead of
    /// letting DuckDB create a path named after it in the repository.
    #[test]
    fn unresolved_placeholders_refuse_the_file() {
        let dir = Path::new("/tmp/x");
        for content in [
            "statement ok\nATTACH 'ducklake:${DUCKLAKE_CONNECTION}' AS d",
            "statement ok\nATTACH 'ducklake:{DUCKLAKE_CONNECTION}' AS d",
            "statement ok\nCOPY t TO '${DATA_PATH}/f.parquet'",
        ] {
            let error = preprocess_test_file(content, dir).err().unwrap();
            assert!(error.contains("unresolved placeholder"), "{error}");
        }
        let resolved = preprocess_test_file(
            "test-env DATA_PATH __TEST_DIR__\n\nstatement ok\nSELECT '${DATA_PATH}', {'a': 1}",
            dir,
        )
        .unwrap();
        assert!(resolved.script.contains("/tmp/x"), "{}", resolved.script);
    }

    /// Numbers compare as numbers only in numeric columns, booleans as
    /// booleans only in boolean columns, and a row must have as many values as
    /// expected.
    /// A `query` header declaring more columns than the query returns fails
    /// the file, as in DuckDB's runner; the right count passes.
    #[tokio::test]
    async fn query_header_column_count_is_enforced() {
        let dir = TempDir::new().unwrap();
        let script = |header: &str| {
            format!(
                "require ducklake\n\nrequire parquet\n\n\
                 test-env DUCKLAKE_CONNECTION __TEST_DIR__/{{UUID}}.db\n\n\
                 test-env DATA_PATH __TEST_DIR__\n\n\
                 statement ok\n\
                 ATTACH 'ducklake:${{DUCKLAKE_CONNECTION}}' AS ducklake \
                 (DATA_PATH '${{DATA_PATH}}/count_files')\n\n\
                 statement ok\nCREATE TABLE ducklake.t (a INTEGER)\n\n\
                 statement ok\nINSERT INTO ducklake.t VALUES (1)\n\n\
                 {header}\nSELECT a FROM ducklake.t\n----\n1\n"
            )
        };
        for (header, passes) in [("query II", false), ("query I", true)] {
            let file = dir.path().join(format!("count_{}.test", header.len()));
            std::fs::write(&file, script(header)).unwrap();
            let outcome = run_hybrid_test(&file).await;
            assert_eq!(
                matches!(outcome, FileOutcome::Passed(_)),
                passes,
                "{header}: {}",
                match outcome {
                    FileOutcome::Failed(error) => error,
                    FileOutcome::Skipped(reason) => format!("skipped: {reason}"),
                    _ => String::new(),
                }
            );
        }
    }

    #[test]
    fn validator_coerces_by_column_kind() {
        use hybrid_asyncdb::{BOOLEAN_TAG, NUMERIC_TAG};
        let row = |values: &[&str]| vec![values.iter().map(|v| v.to_string()).collect()];
        let number = |v: &str| format!("{NUMERIC_TAG}{v}");
        let boolean = |v: &str| format!("{BOOLEAN_TAG}{v}");
        assert!(duckdb_validator(
            &row(&[&number("100")]),
            &["100.0".to_string()]
        ));
        assert!(!duckdb_validator(&row(&["100"]), &["100.0".to_string()]));
        assert!(duckdb_validator(
            &row(&[&boolean("true")]),
            &["1".to_string()]
        ));
        assert!(!duckdb_validator(&row(&["true"]), &["1".to_string()]));
        assert!(!duckdb_validator(&row(&["1"]), &["true".to_string()]));
        // Integers compare exactly, beyond f64's precision.
        assert!(!duckdb_validator(
            &row(&[&number("9007199254740993")]),
            &["9007199254740992".to_string()]
        ));
        assert!(!duckdb_validator(
            &row(&["a", "b"]),
            &["a\tb\tc".to_string()]
        ));
        assert!(!duckdb_validator(
            &row(&["a b", "c"]),
            &["a b c".to_string()]
        ));
        assert!(duckdb_validator(&row(&["a", "b"]), &["a b".to_string()]));
    }

    #[test]
    fn test_statement_error_translation() {
        let lines: Vec<String> = ["statement error", "SELECT 1", "----", "Binder Error: x (y)", ""]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            translate_records(&lines),
            "statement error Binder Error: x \\(y\\)\nSELECT 1\n\n"
        );
    }
}
