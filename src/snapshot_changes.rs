use std::collections::BTreeMap;

use arrow::array::{ListBuilder, MapArray, MapBuilder, StringBuilder};
use datafusion::error::{DataFusionError, Result};
use datafusion::sql::sqlparser::dialect::GenericDialect;
use datafusion::sql::sqlparser::parser::Parser;
use datafusion::sql::sqlparser::tokenizer::Token;

use crate::metadata_provider::snapshot_change_tokens;

pub(crate) fn snapshot_changes<'a>(
    rows: impl Iterator<Item = Option<&'a str>>,
) -> Result<MapArray> {
    let mut builder = MapBuilder::new(
        None,
        StringBuilder::new(),
        ListBuilder::new(StringBuilder::new()),
    );
    for row in rows {
        let mut groups = BTreeMap::<&str, BTreeMap<String, String>>::new();
        let mut schemas = BTreeMap::new();
        for token in row
            .filter(|row| !row.is_empty())
            .into_iter()
            .flat_map(snapshot_change_tokens)
        {
            let (kind, value) = token.split_once(':').ok_or_else(|| invalid_change(token))?;
            let kind = kind.to_ascii_lowercase();
            let key = match kind.as_str() {
                "created_schema" => "schemas_created",
                "created_table" => "tables_created",
                "created_view" => "views_created",
                "created_scalar_macro" => "scalar_macros_created",
                "created_table_macro" => "table_macros_created",
                "dropped_schema" => "schemas_dropped",
                "dropped_table" => "tables_dropped",
                "dropped_view" => "views_dropped",
                "dropped_scalar_macro" => "scalar_macros_dropped",
                "dropped_table_macro" => "table_macros_dropped",
                "altered_table" => "tables_altered",
                "altered_view" => "views_altered",
                "inserted_into_table" => "tables_inserted_into",
                "deleted_from_table" => "tables_deleted_from",
                "inlined_insert" => "inlined_insert",
                "inlined_delete" => "inlined_delete",
                // Existing crate writers record inline_flush
                "flushed_inlined" | "inline_flush" => "flushed_inlined",
                "merge_adjacent" => "merge_adjacent",
                "rewrite_delete" => "rewrite_delete",
                // The official display omits compaction; changes_made retains it
                "compacted_table" => {
                    value.parse::<u64>().map_err(|_| invalid_change(token))?;
                    continue;
                },
                _ => return Err(invalid_change(token)),
            };
            let (sort_key, display) = if kind.starts_with("created_") {
                let mut parser = Parser::new(&GenericDialect)
                    .try_with_sql(value)
                    .map_err(|_| invalid_change(token))?;
                let first = parser
                    .parse_identifier()
                    .map_err(|_| invalid_change(token))?;
                if first.quote_style != Some('"') {
                    return Err(invalid_change(token));
                }
                let display = if kind == "created_schema" {
                    first.value
                } else {
                    parser
                        .expect_token(&Token::Period)
                        .map_err(|_| invalid_change(token))?;
                    let second = parser
                        .parse_identifier()
                        .map_err(|_| invalid_change(token))?;
                    if second.quote_style != Some('"') {
                        return Err(invalid_change(token));
                    }
                    let family = if key == "views_created" {
                        "tables_created"
                    } else {
                        key
                    };
                    let schema = schemas
                        .entry((family, first.value.to_ascii_lowercase()))
                        .or_insert(first.value);
                    format!(
                        "{}.{}",
                        display_identifier(schema),
                        display_identifier(&second.value)
                    )
                };
                if parser.next_token().token != Token::EOF {
                    return Err(invalid_change(token));
                }
                (display.to_ascii_lowercase(), display)
            } else {
                let id = value.parse::<u64>().map_err(|_| invalid_change(token))?;
                (format!("{id:020}"), id.to_string())
            };
            groups
                .entry(key)
                .or_default()
                .entry(sort_key)
                .or_insert(display);
        }
        for (key, values) in groups {
            builder.keys().append_value(key);
            for value in values.into_values() {
                builder.values().values().append_value(value);
            }
            builder.values().append(true);
        }
        builder.append(true)?;
    }
    Ok(builder.finish())
}

fn invalid_change(token: &str) -> DataFusionError {
    DataFusionError::Execution(format!("Invalid DuckLake snapshot change: {token}"))
}

fn display_identifier(name: &str) -> String {
    if name.bytes().enumerate().all(|(index, byte)| {
        byte.is_ascii_alphabetic() || byte == b'_' || (index > 0 && byte.is_ascii_digit())
    }) && !DUCKDB_KEYWORDS
        .split_whitespace()
        .any(|keyword| keyword.eq_ignore_ascii_case(name))
    {
        name.to_string()
    } else {
        format!("\"{}\"", name.replace('"', "\"\""))
    }
}

// DuckDB 1.5.5 quotes every keyword in snapshot object names, including unreserved words
const DUCKDB_KEYWORDS: &str =
    "abort absolute access action add admin after aggregate all also alter always analyse analyze
and anti any array as asc asof assertion assignment asymmetric at attach attribute
authorization backward before begin between bigint binary bit boolean both by cache call called
cascade cascaded case cast catalog centuries century chain char character characteristics check
checkpoint class close cluster coalesce collate collation column columns comment comments
commit committed compression concurrently configuration conflict connection constraint
constraints content continue conversion copy cost create cross csv cube current cursor cycle
data database day days deallocate dec decade decades decimal declare default defaults
deferrable deferred definer delete delimiter delimiters depends desc describe detach dictionary
disable discard distinct do document domain double drop each else enable encoding encrypted end
enum error escape event except exclude excluding exclusive execute exists explain export
export_state extension extensions external extract false family fetch filter first float
following for force foreign forward freeze from full function functions generated glob global
grant granted group grouping grouping_id groups handler having header hold hour hours identity
if ignore ilike immediate immutable implicit import in include including increment index
indexes inherit inherits initially inline inner inout input insensitive insert install instead
int integer intersect interval into invoker is isnull isolation join json key label lambda
language large last lateral leading leakproof left level like limit listen load local location
lock locked logged macro map mapping match matched materialized maxvalue merge method
microsecond microseconds millennia millennium millisecond milliseconds minute minutes minvalue
mode month months move name names national natural nchar new next no none not nothing notify
notnull nowait null nullif nulls numeric object of off offset oids old on only operator option
options or order ordinality others out outer over overlaps overlay overriding owned owner
parallel parser partial partition partitioned passing password percent persistent pivot
pivot_longer pivot_wider placing plans policy position positional pragma preceding precision
prepare prepared preserve primary prior privileges procedural procedure program publication
qualify quarter quarters quote range read real reassign recheck recursive ref references
referencing refresh reindex relative release rename repeatable replace replica reset respect
restart restrict returning returns revoke right role rollback rollup row rows rule sample
savepoint schema schemas scope scroll search second seconds secret security select semi
sequence sequences serializable server session set setof sets share show similar simple skip
smallint snapshot some sorted source sql stable standalone start statement statistics stdin
stdout storage stored strict strip struct subscription substring summarize symmetric sysid
system table tables tablesample tablespace target temp template temporary text then ties time
timestamp to trailing transaction transform treat trigger trim true truncate trusted try_cast
type types unbounded uncommitted unencrypted union unique unknown unlisten unlogged unpack
unpivot until update use user using vacuum valid validate validator value values varchar
variable variadic varying verbose version view views virtual volatile week weeks when where
whitespace window with within without work wrapper write xml xmlattributes xmlconcat xmlelement
xmlexists xmlforest xmlnamespaces xmlparse xmlpi xmlroot xmlserialize xmltable year years yes
zone";

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;
    use rstest::rstest;

    #[rstest]
    #[case("unknown:1")]
    #[case("inserted_into_table:no")]
    #[case("created_table:\"main\"")]
    #[case("created_table:main.events")]
    #[case("created_schema:\"unterminated")]
    #[case("created_schema:\"main\".\"extra\"")]
    #[case("inserted_into_table:1,,deleted_from_table:2")]
    fn malformed_change_is_an_error(#[case] raw: &str) {
        assert!(matches!(
            snapshot_changes([Some(raw)].into_iter()),
            Err(DataFusionError::Execution(_))
        ));
    }

    #[rstest]
    fn absent_and_empty_changes_are_empty_maps() {
        let result = snapshot_changes([None, Some("")].into_iter()).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result.null_count(), 0);
        assert_eq!(result.value_offsets(), &[0, 0, 0]);
    }

    #[rstest]
    fn crate_inline_flush_uses_official_display_key() {
        let actual = snapshot_changes([Some("inline_flush:42")].into_iter()).unwrap();
        let expected = snapshot_changes([Some("flushed_inlined:42")].into_iter()).unwrap();
        assert_eq!(actual, expected);
    }

    #[cfg(feature = "metadata-duckdb")]
    #[rstest]
    fn display_keywords_match_pinned_duckdb() {
        let conn = duckdb::Connection::open_in_memory().unwrap();
        let mut statement = conn
            .prepare("SELECT keyword_name FROM duckdb_keywords() ORDER BY keyword_name")
            .unwrap();
        let keywords = statement
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(
            DUCKDB_KEYWORDS.split_whitespace().collect::<Vec<_>>(),
            keywords
        );
    }
}
