#![cfg(feature = "metadata-duckdb")]

use std::path::Path;
use std::sync::Arc;

use arrow::util::display::{ArrayFormatter, FormatOptions};
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_ducklake::{DuckLakeCatalog, DuckdbMetadataProvider};
use tempfile::TempDir;

fn create_view_catalog(catalog_path: &Path, data_path: &Path) -> anyhow::Result<()> {
    let connection = duckdb::Connection::open_in_memory()?;
    crate::common::ensure_ducklake_installed();
    connection.execute("LOAD ducklake", [])?;
    connection.execute(
        &format!(
            "ATTACH 'ducklake:{}' AS lake (DATA_PATH '{}', DATA_INLINING_ROW_LIMIT 0)",
            catalog_path.display(),
            data_path.display()
        ),
        [],
    )?;
    connection.execute("CREATE TABLE lake.users(id INTEGER, name VARCHAR)", [])?;
    connection.execute("CREATE TABLE lake.scores(id INTEGER, score INTEGER)", [])?;
    connection.execute("CREATE TABLE lake.aliases(id INTEGER, myid INTEGER)", [])?;
    connection.execute("CREATE SCHEMA lake.s1", [])?;
    connection.execute("CREATE TABLE lake.s1.cross_values(value INTEGER)", [])?;
    connection.execute("CREATE SCHEMA lake.MySchema", [])?;
    connection.execute("CREATE TABLE lake.MySchema.mixed_values(value INTEGER)", [])?;
    connection.execute("ATTACH ':memory:' AS ext", [])?;
    connection.execute("CREATE TABLE ext.main.collision(value INTEGER)", [])?;
    connection.execute(
        "INSERT INTO lake.users VALUES (1, 'one'), (2, 'two'), (3, 'three')",
        [],
    )?;
    connection.execute("INSERT INTO lake.scores VALUES (2, 20), (3, 30)", [])?;
    connection.execute("INSERT INTO lake.aliases VALUES (1, 99)", [])?;
    connection.execute("INSERT INTO lake.s1.cross_values VALUES (41)", [])?;
    connection.execute("INSERT INTO lake.MySchema.mixed_values VALUES (42)", [])?;
    connection.execute(
        "CREATE VIEW lake.filtered(identifier, label) AS
         SELECT id, upper(name) FROM lake.users WHERE id > 1",
        [],
    )?;
    connection.execute(
        "CREATE VIEW lake.joined AS
         SELECT users.id, users.name, scores.score
         FROM lake.users JOIN lake.scores USING (id)",
        [],
    )?;
    connection.execute(
        "CREATE VIEW lake.duckdb_only AS SELECT COLUMNS('id') FROM lake.users",
        [],
    )?;
    connection.execute(
        "CREATE VIEW lake.semantics AS
         SELECT id, CASE WHEN score IS NULL THEN -1 ELSE score + 1 END AS adjusted
         FROM lake.scores",
        [],
    )?;
    connection.execute(
        "CREATE VIEW lake.placeholder_collision AS
         SELECT mylake.id FROM lake.aliases AS mylake",
        [],
    )?;
    connection.execute(
        "CREATE VIEW lake.rowid_values AS SELECT rowid FROM lake.users",
        [],
    )?;
    connection.execute(
        "CREATE VIEW lake.rowids AS SELECT rowid FROM lake.rowid_values",
        [],
    )?;
    connection.execute(
        "CREATE VIEW lake.legacy_qualified AS SELECT id FROM lake.users",
        [],
    )?;
    connection.execute("USE lake", [])?;
    connection.execute(
        "CREATE VIEW main.schema_qualified AS SELECT * FROM s1.cross_values",
        [],
    )?;
    connection.execute(
        "CREATE VIEW main.mixed_schema AS SELECT * FROM MySchema.mixed_values",
        [],
    )?;
    connection.execute(
        "CREATE VIEW main.external_qualified AS SELECT * FROM ext.collision",
        [],
    )?;
    connection.execute("CREATE SCHEMA lake.ext", [])?;
    connection.execute("CREATE TABLE lake.ext.collision(value INTEGER)", [])?;
    connection.execute("INSERT INTO lake.ext.collision VALUES (99)", [])?;
    Ok(())
}

fn stored_view_sql(catalog_path: &Path, view_name: &str) -> anyhow::Result<String> {
    let connection = duckdb::Connection::open(catalog_path)?;
    Ok(connection.query_row(
        "SELECT sql FROM ducklake_view WHERE view_name = ?",
        [view_name],
        |row| row.get(0),
    )?)
}

fn normalize_current_view_sql(catalog_path: &Path) -> anyhow::Result<()> {
    let connection = duckdb::Connection::open(catalog_path)?;
    connection.execute(
        "UPDATE ducklake_view
         SET sql = CASE
             WHEN view_name = 'legacy_qualified'
                 THEN replace(sql, '{DUCKLAKE_CATALOG}.', 'lake.')
             ELSE replace(sql, 'lake.', '{DUCKLAKE_CATALOG}.')
         END",
        [],
    )?;
    Ok(())
}

fn duckdb_rows(
    catalog_path: &Path,
    data_path: &Path,
    sql: &str,
) -> anyhow::Result<Vec<Vec<String>>> {
    let connection = duckdb::Connection::open_in_memory()?;
    connection.execute("LOAD ducklake", [])?;
    connection.execute(
        &format!(
            "ATTACH 'ducklake:{}' AS lake (DATA_PATH '{}')",
            catalog_path.display(),
            data_path.display()
        ),
        [],
    )?;
    let mut statement = connection.prepare(sql)?;
    let rows = statement
        .query_map([], |row| {
            (0..row.as_ref().column_count())
                .map(|column| row.get::<_, String>(column))
                .collect::<Result<Vec<_>, _>>()
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

async fn datafusion_rows(context: &SessionContext, sql: &str) -> anyhow::Result<Vec<Vec<String>>> {
    let batches = context.sql(sql).await?.collect().await?;
    let options = FormatOptions::default();
    let mut rows = Vec::new();
    for batch in batches {
        let formatters = batch
            .columns()
            .iter()
            .map(|array| ArrayFormatter::try_new(array.as_ref(), &options))
            .collect::<Result<Vec<_>, _>>()?;
        for row in 0..batch.num_rows() {
            rows.push(
                formatters
                    .iter()
                    .map(|formatter| formatter.value(row).to_string())
                    .collect(),
            );
        }
    }
    Ok(rows)
}

#[tokio::test]
async fn duckdb_views_are_listed_and_queryable() -> anyhow::Result<()> {
    let temp = TempDir::new()?;
    let catalog_path = temp.path().join("views.ducklake");
    let data_path = temp.path().join("data");
    create_view_catalog(&catalog_path, &data_path)?;
    let oracle_queries = [
        "SELECT CAST(identifier AS VARCHAR) AS identifier_text, label
         FROM lake.filtered ORDER BY identifier",
        "SELECT CAST(id AS VARCHAR) AS id_text, name, CAST(score AS VARCHAR) AS score_text
         FROM lake.joined ORDER BY id",
        "SELECT CAST(id AS VARCHAR) AS id_text, CAST(adjusted AS VARCHAR) AS adjusted_text
         FROM lake.semantics ORDER BY id",
        "SELECT CAST(value AS VARCHAR) AS value_text FROM lake.schema_qualified",
        "SELECT CAST(value AS VARCHAR) AS value_text FROM lake.mixed_schema",
    ];
    let oracle_rows = oracle_queries
        .iter()
        .map(|query| duckdb_rows(&catalog_path, &data_path, query))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let rowid_oracle = duckdb_rows(
        &catalog_path,
        &data_path,
        "SELECT CAST(rowid AS VARCHAR) AS rowid_text FROM lake.rowids ORDER BY rowid",
    )?;
    assert!(stored_view_sql(&catalog_path, "filtered")?.contains("{DUCKLAKE_CATALOG}.users"));
    normalize_current_view_sql(&catalog_path)?;
    assert!(stored_view_sql(&catalog_path, "filtered")?.contains("{DUCKLAKE_CATALOG}.users"));
    assert!(stored_view_sql(&catalog_path, "legacy_qualified")?.contains("lake.users"));
    assert!(stored_view_sql(&catalog_path, "schema_qualified")?.contains("s1.cross_values"));
    assert!(stored_view_sql(&catalog_path, "mixed_schema")?.contains("MySchema.mixed_values"));
    assert!(stored_view_sql(&catalog_path, "external_qualified")?.contains("ext.collision"));

    let provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy())?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let context = SessionContext::new();
    context.register_catalog("ducklake", Arc::new(catalog));

    let schema = context.catalog("ducklake").unwrap().schema("main").unwrap();
    assert_eq!(
        schema.table_names(),
        vec![
            "aliases",
            "duckdb_only",
            "external_qualified",
            "filtered",
            "joined",
            "legacy_qualified",
            "mixed_schema",
            "placeholder_collision",
            "rowid_values",
            "rowids",
            "schema_qualified",
            "scores",
            "semantics",
            "users",
        ]
    );
    assert!(schema.table_exist("filtered"));

    context
        .sql("SELECT * FROM ducklake.main.filtered")
        .await?
        .collect()
        .await?;

    for (query, expected) in oracle_queries.into_iter().zip(oracle_rows) {
        let datafusion_query = query.replace("lake.", "ducklake.main.");
        assert_eq!(
            datafusion_rows(&context, &datafusion_query).await?,
            expected
        );
    }

    let views = datafusion_rows(
        &context,
        "SELECT schema_name, view_name, dialect, column_aliases
         FROM ducklake.information_schema.views
         ORDER BY view_name",
    )
    .await?;
    assert_eq!(
        views,
        vec![
            vec!["main", "duckdb_only", "duckdb", ""],
            vec!["main", "external_qualified", "duckdb", ""],
            vec!["main", "filtered", "duckdb", "\"identifier\",\"label\""],
            vec!["main", "joined", "duckdb", ""],
            vec!["main", "legacy_qualified", "duckdb", ""],
            vec!["main", "mixed_schema", "duckdb", ""],
            vec!["main", "placeholder_collision", "duckdb", ""],
            vec!["main", "rowid_values", "duckdb", ""],
            vec!["main", "rowids", "duckdb", ""],
            vec!["main", "schema_qualified", "duckdb", ""],
            vec!["main", "semantics", "duckdb", ""],
        ]
    );

    let error = context
        .sql("SELECT * FROM ducklake.main.duckdb_only")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("DuckLake view 'duckdb_only'"), "{error}");
    assert!(error.contains("dialect 'duckdb'"), "{error}");

    let error = context
        .sql("SELECT * FROM ducklake.main.placeholder_collision")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("DuckLake view 'placeholder_collision'"),
        "{error}"
    );

    let error = context
        .sql("SELECT * FROM ducklake.main.legacy_qualified")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("DuckLake view 'legacy_qualified'"),
        "{error}"
    );

    let error = context
        .sql("SELECT * FROM ducklake.main.external_qualified")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("DuckLake view 'external_qualified'"),
        "{error}"
    );

    let enumeration_provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy())?;
    let enumeration_catalog = DuckLakeCatalog::new(enumeration_provider)?;
    let enumeration_context =
        SessionContext::new_with_config(SessionConfig::new().with_information_schema(true));
    enumeration_context.register_catalog("ducklake", Arc::new(enumeration_catalog));

    let built_in_views = datafusion_rows(
        &enumeration_context,
        "SELECT table_name, definition
         FROM information_schema.views
         WHERE table_catalog = 'ducklake' AND table_schema = 'main'
         ORDER BY table_name",
    )
    .await?;
    assert_eq!(built_in_views.len(), 14);
    let definition = |name: &str| {
        built_in_views
            .iter()
            .find(|row| row[0] == name)
            .map(|row| row[1].as_str())
            .unwrap()
    };
    assert!(!definition("filtered").contains("{DUCKLAKE_CATALOG}"));
    assert!(definition("placeholder_collision").contains("{DUCKLAKE_CATALOG}"));

    datafusion_rows(
        &enumeration_context,
        "SELECT table_name, column_name
         FROM information_schema.columns
         WHERE table_catalog = 'ducklake' AND table_schema = 'main'
         ORDER BY table_name, ordinal_position",
    )
    .await?;

    let lineage_provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy())?;
    let lineage_catalog = DuckLakeCatalog::new(lineage_provider)?.with_row_lineage(true);
    let lineage_context = SessionContext::new();
    lineage_context.register_catalog("ducklake", Arc::new(lineage_catalog));
    assert_eq!(
        datafusion_rows(
            &lineage_context,
            "SELECT CAST(rowid AS VARCHAR) AS rowid_text FROM ducklake.main.rowids ORDER BY rowid",
        )
        .await?,
        rowid_oracle
    );
    Ok(())
}
