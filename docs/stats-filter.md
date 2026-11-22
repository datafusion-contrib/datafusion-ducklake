# Statistics pruning in the catalog query

`src/stats_filter.rs` lowers a pushed-down predicate into SQL that narrows the data-file
listing against `ducklake_file_column_stats`, as official DuckLake does in
`ConvertFilterPushdownToSQL` (`ducklake_metadata_manager.cpp`). This page records where that
rewrite deliberately differs from official, and why.

At the pinned oracle `d8a1881e`, official tests CTE membership twice
(`data_file_id NOT IN (SELECT data_file_id FROM <cte>) OR data_file_id IN (SELECT ... WHERE
<condition>)`), so it references the CTE twice. This crate names the CTE once, in a
`LEFT JOIN`, and reads the no-stats-row case off a NULL `data_file_id`. Upstream moved to a
`LEFT JOIN` itself in `71a9e105`, after the oracle. The reference count matters to *Why the
statistics CTE is materialized* below.

## Deliberate deviations from official DuckLake

Every one of these is a divergence, and the list is meant to be exhaustive —
a shorter list than the code has is worse than no list. Each either follows
from where the SQL runs, or from how this engine orders values. None of them
prunes a file official would keep except where noted.

**1. `TRY_CAST` is not portable, so each dialect supplies its own.** Official
always evaluates its filter SQL inside DuckDB, even against a foreign
catalog: its Postgres manager overrides only the CTE *body* and wraps it in
`postgres_query(...)`, so the join, the casts and the predicate all run in
DuckDB. This crate sends SQL natively to each catalog engine, where
`TRY_CAST` does not exist and a plain `CAST` of a malformed stat either
aborts the query (PostgreSQL) or silently returns zero (SQLite, MySQL) — the
second of which would prune a matching file. Each dialect supplies a
NULL-on-unparseable construct through `StatsSqlDialect::try_cast`, and may
decline a type it cannot handle safely, which drops that comparison. The
same split decides CTE materialization: the planner that receives this SQL
is the catalog's, not DuckDB's, so how the statistics CTE is declared is
this crate's decision to make and its dialects' to spell — see *Why the
statistics CTE is materialized*.

**2. Raw string comparisons force a binary collation.** Official inherits
DuckDB's byte-wise collation. A native MySQL catalog defaults to
`utf8mb4_0900_ai_ci`, which is case- and accent-insensitive and would drop
files that match. DataFusion compares `Utf8` byte-wise, so raw comparisons go
through `StatsSqlDialect::collate_binary`.

**3. Float bounds are gated on `contains_nan`, except for equality.**
Catalog bounds for a float column exclude NaN, so a file whose NaN state is
unknown or positive can hold values outside them. DuckDB normalizes NaN —
`-NaN = NaN` is true and either sign sorts above every value — so there NaN
can only hide *above* a recorded max, and official guards only the max
(`GenerateConstantFilterDouble` appends `OR contains_nan` for `>`, `>=` and
`<>`). DataFusion compares floats with arrow's `total_cmp`, which is
sign-sensitive: `-NaN < -Infinity` is true here and false in DuckDB. A
negative NaN therefore sits *below* a recorded min, and `contains_nan` does
not record the sign, so neither bound can be trusted.
`float_bound_is_usable` in `table.rs` gates both bounds for that reason, and
`nan_pruning_barrier` keeps such predicates out of the parquet scan
for the same one. So a condition reading a bound on a float column is
rendered as `(<bounds not usable>) OR (<official condition>)`. That matches
official exactly for `>`, `>=` and `<>`, where its own `OR contains_nan`
already keeps every NaN-bearing file, and is strictly more conservative for
`<` and `<=`. Equality is left ungated, as official leaves it: no NaN
compares equal to a finite value under `total_cmp`, and
`StatsLiteral::new` refuses a NaN constant, so every row satisfying
`x = C` is a non-NaN row and the recorded bounds already decide it.

**4. A stat is validated against its encoding before it is used.** Official
hands the stored text straight to `TRY_CAST`. That is safe for official
because DuckDB wrote every stat it reads; a catalog this crate opens may have
been written by anything, including `ducklake_add_files` over a pre-1.11
parquet-mr file whose float min/max hold NaN. An input function is
permissive by design: DuckDB reads `nan`, `epoch` and `0x10` into real
values, and PostgreSQL's `pg_input_is_valid` additionally accepts `today`,
`now`, `infinity` and `NaN`. Each casts to something that then prunes files —
a `nan` bound makes `x = 5.0` false for a file whose other rows match, and a
`today` bound would make pruning depend on the wall clock. So every dialect
admits only the shapes `crate::stats_encode` writes. This is strictly more
conservative than official and declines a few stats official would accept.

**5. A condition that evaluates to SQL `NULL` keeps its file.** Official has
no counterpart; `StatsSqlDialect::keep_when_unknown` wraps every per-column
condition. The per-stat `IS NULL` disjuncts cannot cover this on their own,
because a stat that is present but malformed is not NULL while the cast of it
is, and the condition sits under `WHERE ... AND`, where `NULL` excludes the
row. Pruning happens only on a definite `false`.

**6. Temporal comparisons happen in the text domain on some dialects.**
Official casts them. PostgreSQL and MySQL store microseconds and *round* a
longer fraction, which is monotonic but not injective — two distinct
nanosecond instants can land on one microsecond, so a strict comparison that
holds of the stored values comes back false. SQLite has no temporal type at
all. The canonical encoding is chronologically ordered byte-wise, so
comparing the two strings answers the same question exactly, at full
precision, and parses nothing — an impossible date cannot raise. It is sound
only for that one encoding with a four-digit year: a year past 9999 has five
digits (`12345-01-01` sorts below `2024-01-01`), one before the common era
is marked ` (BC)` after the date, and `.50` sorts above `.5` while naming the
same instant. Anything else is declined.

**7. A NaN or non-finite constant pushes down nothing.** Official prunes on
`contains_nan` for `x = NaN`, and emits a real comparison for `x < inf`.
`StatsLiteral::new` refuses both: a non-finite constant renders as a quoted
literal, which a dialect that coerces text to numbers reads as zero, and
`x < inf` would then prune every file with a non-negative minimum. This is
less pruning than official, never more.

**8. The float gate keys off the column's type, official off the
constant's.** Official's NaN handling is selected by the constant
(`type ? *type : constant_expr->GetValue().type()`, with `type` always null
from its entry point). It is the *column's* stored bounds that exclude NaN,
and a `PhysicalExpr` handed to `crate::DuckLakeTable::files_matching`
carries no guarantee the constant was coerced to the column's type, so the
gate follows the column while the cast still follows the constant. A
constant whose type differs from a float column's, or a float constant on a
column of another type, pushes down nothing: the cast would round a bound,
and a rounded bound can land on the constant and fail a strict comparison
its file satisfies (an `INTEGER` past 2^53 read as a `DOUBLE`). The same
holds for a `DECIMAL` constant of smaller scale than its column. Official
casts there and can prune a matching file; this keeps it.

**9. A dialect's conversion of a bound is computed once per file, in the CTE
body.** *A consequence of divergence 1, not an independent choice: this would
not exist if the filter ran in DuckDB.* Official's
`GenerateFileColumnStatsCTEBody` emits a plain projection of the stored text
and puts the cast in the conditions (`CastStatsToTarget`), so an `IN` list of
n values converts the same two strings n times per file. That is nearly free
in DuckDB, where the condition runs over a vectorized column, and it is not
free in a catalog engine evaluating it row by row — and it is why
materializing the CTE (below) does not fix it on its own: what gets
materialized is the raw text, while the conversion lives in the join filter.
Each conversion this crate needs is therefore emitted once as a CTE output
column and the condition names that column. Measured on PostgreSQL 18 over a
synthetic catalog of 8192 files carrying one statistics row each, with a
32-value `IN` predicate: **roughly twenty times faster**, 2703 ms with the
conversion in the join filter against 127 ms with it in the CTE body, the
same 32 rows from both. Read the ratio, not the milliseconds — every figure
in this module is bound to that fixture, and an independent rebuild of the
same shape reproduced the direction with different magnitudes. Results are
unchanged, and
structurally so: the raw statistics stay in the projection because the
fail-open `IS NULL` disjuncts read them, so the statement is the previous one
with each conversion replaced by a reference to itself, and inlining the CTE
reproduces it verbatim.

**10. An identity partition key pre-filters the listing, as a bucket key does
in official.** *Also a consequence of divergence 1: in DuckDB the statistics
path already costs almost nothing, so official never needed this for an
identity key.* Official narrows the same listing against
`ducklake_file_partition_value` (`BuildBucketPartitionPruningClause`), and
only for `BUCKET` transforms — `if (field.transform.type !=
DuckLakeTransformType::BUCKET) continue;` — because hashing destroys ordering
so statistics cannot prune a bucket partition at all, while for `identity`
the statistics path suffices *in DuckDB*. It does not suffice here: the
statistics path costs one conversion per file however few files a query
probes. Extending official's own mechanism to the transform it did not need
it for measured, on the same fixture and with the same caveat: **roughly
twenty times again**, 127 ms with divergence 9 alone against 6.8 ms with the
pre-filter. The conjunct goes on the outer `WHERE` *and* inside each
statistics CTE body, because a materialized CTE is an optimization fence the
outer conjunct cannot reach into: with it only on the outer `WHERE` the same
query still takes 122 ms, which is no gain at all.

The comparison is on the stored *text*, not on a converted value, and that is
the whole of why it is fast: converting the partition value instead lands
back at 117 ms, the unfiltered listing again. Text comparison is sound only
while no other spelling of a listed value exists, so four things narrow it,
all of them keeping files official's clause would drop. It is an anti-join
rather than `data_file_id IN (...)`, so a file with no row for the key is
kept. A NULL partition value is kept. Only types whose value has one
rendering are pushed down at all — integer and string columns
(`PartitionTextMatch`); a float, decimal, boolean or temporal key pre-filters
nothing. And an integer value refutes only once
`StatsSqlDialect::canonical_integer_text` has *positively* proven the
stored text is the canonical spelling, because integer parsers are lenient in
ways no blocklist covers — DuckDB reads `2.0`, `0x2`, `2e0` and
whitespace-padded text all as 2, and each of those must keep its file. A
dialect with no way to prove that declines, and the key pre-filters nothing.
Official folds its constant through DuckDB and compares `Value::ToString()`;
this crate compares against the encoding its own write path stored
(`split_batches_by_partition` calls `stats_encode::encode_scalar`, the same
function that produces `StatsLiteral::text`).

**11. A `FLOAT` constant is compared in the double domain, through the range
of doubles that name it.** Official casts a `FLOAT` bound with
`TRY_CAST(... AS FLOAT)` and splices the constant as its shortest text,
which DuckDB reads as a `DOUBLE` whenever it carries an exponent: `1e+20` is
10^20 as a double, while the `FLOAT` closest to it, widened, is
1.0000000200408773e+20. The comparison then runs in `DOUBLE` and drops the
file holding the very row `c = 1e20::FLOAT` asks for (duckdb/ducklake#1604).
Reading both sides as text-to-double does not fix it in general: the
shortest spelling of a float is one of many, and another writer's
`1.00000002e+20` reads as a different double. Every correctly rounded
spelling of a float does read as a double between the two midpoints to its
neighbours, so the bound is cast to the dialect's double type and each
comparison is made against a midpoint instead of the constant — `min < C`
becomes `min <= lo(C)`, `max >= C` becomes `max >= lo(C)`, and so on, see
`FloatWindow`. Every engine has a double type, so this is one rendering for
all four dialects. A bound that lies exactly on a midpoint could name either
neighbour, so every edge comparison is non-strict; a file is kept that a
float-domain comparison would drop only when a bound sits exactly there.
A zero constant, `FLOAT` or `DOUBLE`, gets the window around both zeros.
DataFusion's `total_cmp` orders `-0.0` below `0.0`, so `c < 0.0` matches a
`-0.0` row and `c <> 0.0` a `-0.0` one; the catalog engines compare the two
zeros equal, and a bound need not carry the sign of the zero it stands for
(a writer records a largest `-0.0` as `0.0`, as the Parquet specification
has it). So `<` and `>` against a zero keep every file with a zero bound,
and `<>` pushes down nothing. A `DOUBLE` constant other than zero is compared exactly, as
official does, but spliced as its shortest digits with an exponent
(`4.8024702758086213e-1`). Official splices its shortest text, which for a
magnitude DuckDB prints without an exponent is fixed-point
(`0.48024702758086213`); DuckDB parses that as a `DECIMAL`, and its
`DECIMAL` to `DOUBLE` conversion is not correctly rounded past 15 digits,
so the constant reads as a neighbouring double and official drops the file
holding the very row `d = 0.48024702758086213` asks for. A literal with an
exponent is a `DOUBLE` to DuckDB and MySQL, and a numeric that converts to
the correctly rounded double on PostgreSQL; SQLite reads both spellings
alike.

## Why the statistics CTE is materialized

The CTE selects the raw `min_value` / `max_value` text and every comparison
against it carries the dialect's validity test and its cast. Inlined, that
work lands in the join condition and is redone once per comparison per file,
so an `IN` list of n values reads the same two strings n times over — the
cost of pruning then grows with the *product* of the table's file count and
the predicate's width, which is invisible on a small table and dominates
planning on a large partitioned one. `StatsSqlDialect::cte_materialization`
asks for one evaluation per file instead.

Official materializes the same CTE at oracle commit `d8a1881e`, under the
same `col_<n>_stats` alias, so the shape converges. Two things differ. The
first is which planner reads the hint: official's filter SQL runs in DuckDB
whatever the catalog is, and this crate's runs in the catalog engine, which
is divergence 1 again and is why the spelling has to come from the dialect
at all — one of the four engines has no such modifier.

The second is how the decision is reached, and it is why this is a cost
model rather than a port. Official chooses on a reference count
(`reference_count > 1` in `GenerateCTESectionFromRequirements`), which is a
sound proxy for *its* shape: its CTE body is a plain projection, all the
casting lives in the conditions, and it names the CTE twice — once in a
`NOT IN` and once in an `IN`. This crate names it once, in a `LEFT JOIN`, so
official's own rule applied here would emit `AS NOT MATERIALIZED` and ask
for precisely the inlining the paragraph above exists to avoid. A reference
count cannot see that, because the expense is not how often the CTE is read
but that each read re-runs a validity test and a cast the count knows
nothing about.

Upstream has since moved: `71a9e105` replaced the double membership test
with a `LEFT JOIN` (which drops the reference count to 1), and `f67a2bd7`
removed the hint machinery outright 28 minutes later, leaving the choice to
DuckDB. Both land after the pinned oracle. **When the oracle moves past
2026-08-29, official no longer hints, and this stops being convergence and
becomes a divergence to re-argue** — on the measured-performance limb, since
DuckDB's defaults were never chosen for a native catalog's planner.
Re-adjudicate it then rather than meeting it as drift.
