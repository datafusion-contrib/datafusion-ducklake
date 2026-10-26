//! Live row counts per partition value, from catalog metadata alone.
//!
//! [`DuckLakeTable::partition_live_row_counts`](crate::DuckLakeTable::partition_live_row_counts)
//! answers "how many rows are live in the partition whose value is `v`" without
//! listing the table's files to the caller or opening any of them. A provider
//! supplies the catalog facts ([`PartitionCountFacts`], through
//! [`MetadataProvider::get_partition_count_facts`](crate::MetadataProvider::get_partition_count_facts)),
//! and the table turns them into a count per requested value.
//!
//! The answer for a value is [`PartitionRowCount::Exact`] only when the catalog
//! counters account for every row a scan at the same snapshot would return, and
//! [`PartitionRowCount::Unknown`] otherwise. Unknown is never zero: a caller
//! that needs the number has to count the partition's rows for real.
//!
//! The column must be the one identity key of the table's live partition spec,
//! and each requested value must be spelled as the writer stores it (see
//! [`PartitionCountUnknown::NonCanonicalValue`]). A value is then exact when all
//! of these hold at the read snapshot `S`:
//!
//! - `S` still exists in the catalog;
//! - the table has no visible inlined rows. Inlined rows live in the catalog,
//!   not in a data file, and no file counter includes them. This is stricter
//!   than official DuckLake, whose nearest mechanism (the `count(*)` fold of
//!   `ducklake_scan.cpp`, over `GetNetDataFileRowCountSql`) is table-wide and
//!   adds the inlined rows in; per partition value that would mean reading
//!   every inlined row's key, so here any visible inlined row makes every
//!   value unknown;
//! - every visible data file is attributable: it was written under the live
//!   partition spec, has exactly one partition value for the key, and that
//!   value is spelled canonically. One that is not could hold rows of any
//!   value, so every value is unknown;
//! - every file of the value records a `record_count`;
//! - no file of the value is a merged partial file read below its
//!   `partial_max`, whose rows from later snapshots a scan drops one by one;
//! - each file of the value has at most one visible delete file, which records
//!   a `delete_count` between 0 and the file's `record_count`, and whose
//!   deletions were all made by `S`. A delete file can begin before some of its
//!   deletions: a delete file that rewrites an earlier one records the latest
//!   deletion as `partial_max`, but a flush of inlined rows deleted while
//!   inlined writes one that begins at its earliest deletion and records no
//!   `partial_max` (official's `ducklake_flush_inlined_data` does the same). So a delete file counts
//!   below the catalog head only with a `partial_max` at or below `S`; at the
//!   head every deletion it holds has been made, as official answers
//!   `count(*)` from delete counts only at the current snapshot;
//! - a file of the value with inlined deletions has no delete file. The scan
//!   applies the union of the two, and only reading the delete file could say
//!   whether they overlap;
//! - the sum fits in a `u64`.
//!
//! A requested value with no visible files is exact at zero. A catalog that
//! predates `ducklake_delete_file.partial_max` (or the partition tables)
//! answers [`PartitionCountUnknown::Unsupported`].

use std::collections::{BTreeMap, HashMap, HashSet};

use arrow::datatypes::DataType;

use crate::metadata_provider::DuckLakeInlinedDelete;

/// The live row count of one partition value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PartitionRowCount {
    /// The catalog accounts for every live row of the value.
    Exact(u64),
    /// The catalog cannot give an exact answer; count the rows instead.
    Unknown(PartitionCountUnknown),
}

impl PartitionRowCount {
    /// The count, when it is exact.
    pub fn exact(self) -> Option<u64> {
        match self {
            Self::Exact(rows) => Some(rows),
            Self::Unknown(_) => None,
        }
    }
}

/// Why a partition value's count is unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PartitionCountUnknown {
    /// The metadata provider does not read partition counts, or the catalog
    /// lacks a table or column the count needs.
    Unsupported,
    /// The column is not a key of the table's partition spec at the snapshot.
    NotAPartitionKey,
    /// The column is a key of the spec only through a transform (`year(ts)`,
    /// `bucket(4, id)`), whose values are not the column's.
    NotIdentityKey,
    /// The column is more than one key of the spec.
    AmbiguousKey,
    /// The key is a floating-point column. Its partitions are spelled values,
    /// and `0.0` and `-0.0` are two of them although they compare equal, so a
    /// partition's count is not the count of rows equal to its value.
    FloatKey,
    /// The requested value is not spelled as the writer stores a value of the
    /// key's type (`"02"` or `"+2"` for the integer 2), so no stored value
    /// would match it as text.
    NonCanonicalValue,
    /// The table's snapshot no longer exists in the catalog.
    SnapshotMissing,
    /// A visible data file was written under another partition spec, has no
    /// partition value (or more than one) for the key, or has a value not
    /// spelled canonically, so it cannot be attributed to a value.
    UnattributedFile,
    /// The table has visible inlined rows, which no file counter includes.
    InlinedRows,
    /// A data file of the value records no valid `record_count`.
    MissingRecordCount,
    /// A data file of the value is a merged partial file read below its
    /// `partial_max`.
    PartialFile,
    /// A data file of the value has more than one visible delete file.
    MultipleDeleteFiles,
    /// A delete file of the value records no `delete_count`, or one outside
    /// `0..=record_count`.
    InvalidDeleteCount,
    /// A delete file of the value may hold deletions made after the snapshot.
    PartialDeleteFile,
    /// A data file of the value has both a delete file and inlined deletions.
    InlinedDeletesWithDeleteFile,
    /// The count does not fit in a `u64`.
    Overflow,
}

/// Catalog facts for a partition count, as a provider reads them at one
/// snapshot. Build one with [`Self::new`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct PartitionCountFacts {
    /// Whether the snapshot exists in `ducklake_snapshot`.
    pub snapshot_exists: bool,
    /// Whether any inlined-data table of the table holds a row visible at the
    /// snapshot. A provider that cannot tell reports `true`.
    pub inlined_rows_visible: bool,
    /// Every distinct non-NULL value any visible data file has for the key.
    pub stored_values: Vec<String>,
    /// Every visible data file whose partition value for the key is one of the
    /// requested values, and every visible data file that cannot be attributed
    /// to a value (see [`PartitionCountFile::partition_id`] and
    /// [`PartitionCountFile::value_rows`]). A provider may return other visible
    /// files too; they are ignored.
    pub files: Vec<PartitionCountFile>,
}

impl PartitionCountFacts {
    /// Facts read at a snapshot that exists (`snapshot_exists`), with or
    /// without visible inlined rows, the key's distinct stored values, and the
    /// files described on [`Self::files`].
    pub fn new(
        snapshot_exists: bool,
        inlined_rows_visible: bool,
        stored_values: Vec<String>,
        files: Vec<PartitionCountFile>,
    ) -> Self {
        Self {
            snapshot_exists,
            inlined_rows_visible,
            stored_values,
            files,
        }
    }
}

/// One visible data file, for a partition count. Build one with
/// [`Self::new`] and, when it has visible delete files,
/// [`Self::with_delete_files`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct PartitionCountFile {
    /// `ducklake_data_file.data_file_id`.
    pub data_file_id: i64,
    /// `ducklake_data_file.partition_id`.
    pub partition_id: Option<i64>,
    /// How many `ducklake_file_partition_value` rows the file has for the key.
    pub value_rows: i64,
    /// The file's value for the key, when it has exactly one.
    pub partition_value: Option<String>,
    /// `ducklake_data_file.record_count`.
    pub record_count: Option<i64>,
    /// `ducklake_data_file.partial_max`.
    pub partial_max: Option<i64>,
    /// How many delete files of the file are visible at the snapshot.
    pub delete_files: i64,
    /// The visible delete file's `delete_count`, when there is exactly one.
    pub delete_count: Option<i64>,
    /// The visible delete file's `partial_max`, when there is exactly one.
    pub delete_partial_max: Option<i64>,
}

impl PartitionCountFile {
    /// A data file with no visible delete file.
    pub fn new(
        data_file_id: i64,
        partition_id: Option<i64>,
        value_rows: i64,
        partition_value: Option<String>,
        record_count: Option<i64>,
        partial_max: Option<i64>,
    ) -> Self {
        Self {
            data_file_id,
            partition_id,
            value_rows,
            partition_value,
            record_count,
            partial_max,
            ..Self::default()
        }
    }

    /// The file with `delete_files` visible delete files, and the
    /// `delete_count` and `partial_max` of the one when there is one.
    pub fn with_delete_files(
        mut self,
        delete_files: i64,
        delete_count: Option<i64>,
        delete_partial_max: Option<i64>,
    ) -> Self {
        self.delete_files = delete_files;
        self.delete_count = delete_count;
        self.delete_partial_max = delete_partial_max;
        self
    }
}

/// Whether `value` is spelled as the writer stores a partition value of
/// `data_type`: a string verbatim, anything else as
/// [`crate::stats_encode::encode_scalar`] spells the value it parses to.
pub(crate) fn is_canonical_partition_value(value: &str, data_type: &DataType) -> bool {
    if matches!(
        data_type,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    ) {
        return true;
    }
    crate::types::parse_ducklake_scalar(value, data_type)
        .and_then(|scalar| crate::stats_encode::encode_scalar(&scalar))
        .is_some_and(|encoded| encoded == value)
}

/// What [`resolve_partition_counts`] counts at.
pub(crate) struct CountRead<'a> {
    /// The table's snapshot.
    pub(crate) snapshot_id: i64,
    /// Whether that snapshot is the catalog's head.
    pub(crate) at_head: bool,
    /// The live partition spec's id.
    pub(crate) partition_id: i64,
    /// The key column's type.
    pub(crate) key_type: &'a DataType,
}

/// The live row count of each of `values`, from `facts` and the table's
/// `inlined_deletes` read at `read`'s snapshot. `facts` of `None` means the
/// provider cannot answer.
pub(crate) fn resolve_partition_counts(
    values: &[String],
    read: &CountRead<'_>,
    facts: Option<&PartitionCountFacts>,
    inlined_deletes: &[DuckLakeInlinedDelete],
) -> BTreeMap<String, PartitionRowCount> {
    let all = |reason| {
        values
            .iter()
            .map(|value| (value.clone(), PartitionRowCount::Unknown(reason)))
            .collect()
    };
    let Some(facts) = facts else {
        return all(PartitionCountUnknown::Unsupported);
    };
    if !facts.snapshot_exists {
        return all(PartitionCountUnknown::SnapshotMissing);
    }
    if facts.inlined_rows_visible {
        return all(PartitionCountUnknown::InlinedRows);
    }
    if facts
        .files
        .iter()
        .any(|file| file.partition_id != Some(read.partition_id) || file.value_rows != 1)
        || facts
            .stored_values
            .iter()
            .any(|value| !is_canonical_partition_value(value, read.key_type))
    {
        return all(PartitionCountUnknown::UnattributedFile);
    }
    let mut inlined_by_file: HashMap<i64, HashSet<i64>> = HashMap::new();
    for delete in inlined_deletes {
        inlined_by_file
            .entry(delete.data_file_id)
            .or_default()
            .insert(delete.row_id);
    }
    let mut counts: BTreeMap<String, PartitionRowCount> = values
        .iter()
        .map(|value| {
            let count = if is_canonical_partition_value(value, read.key_type) {
                PartitionRowCount::Exact(0)
            } else {
                PartitionRowCount::Unknown(PartitionCountUnknown::NonCanonicalValue)
            };
            (value.clone(), count)
        })
        .collect();
    let mut seen = HashSet::new();
    for file in &facts.files {
        let Some(value) = file.partition_value.as_ref() else {
            continue;
        };
        let Some(count) = counts.get_mut(value) else {
            continue;
        };
        if !seen.insert(file.data_file_id) {
            continue;
        }
        let PartitionRowCount::Exact(total) = *count else {
            continue;
        };
        *count = match live_rows(file, read, inlined_by_file.get(&file.data_file_id)) {
            Ok(live) => match total.checked_add(live) {
                Some(sum) => PartitionRowCount::Exact(sum),
                None => PartitionRowCount::Unknown(PartitionCountUnknown::Overflow),
            },
            Err(reason) => PartitionRowCount::Unknown(reason),
        };
    }
    counts
}

/// The live rows of one attributable `file` at `read`'s snapshot, given the
/// positions of its inlined deletions.
fn live_rows(
    file: &PartitionCountFile,
    read: &CountRead<'_>,
    inlined: Option<&HashSet<i64>>,
) -> std::result::Result<u64, PartitionCountUnknown> {
    let rows = match file.record_count {
        Some(rows) if rows >= 0 => rows,
        _ => return Err(PartitionCountUnknown::MissingRecordCount),
    };
    if file.partial_max.is_some_and(|max| read.snapshot_id < max) {
        return Err(PartitionCountUnknown::PartialFile);
    }
    let deleted = match file.delete_files {
        0 => 0,
        1 => {
            let complete = match file.delete_partial_max {
                Some(max) => max <= read.snapshot_id,
                None => read.at_head,
            };
            if !complete {
                return Err(PartitionCountUnknown::PartialDeleteFile);
            }
            match file.delete_count {
                Some(deleted) if (0..=rows).contains(&deleted) => deleted,
                _ => return Err(PartitionCountUnknown::InvalidDeleteCount),
            }
        },
        _ => return Err(PartitionCountUnknown::MultipleDeleteFiles),
    };
    // As the scan applies them: a set, and only positions the file holds.
    let inlined = inlined.map_or(0, |positions| {
        positions
            .iter()
            .filter(|position| (0..rows).contains(*position))
            .count() as i64
    });
    if inlined > 0 && file.delete_files > 0 {
        return Err(PartitionCountUnknown::InlinedDeletesWithDeleteFile);
    }
    // `deleted` is within `0..=rows`, and `inlined` is too, and only one of
    // them is non-zero.
    Ok(u64::try_from(rows - deleted - inlined).unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPEC: i64 = 7;

    fn file(id: i64, value: &str, rows: i64) -> PartitionCountFile {
        PartitionCountFile::new(id, Some(SPEC), 1, Some(value.to_string()), Some(rows), None)
    }

    fn values(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    fn facts(files: Vec<PartitionCountFile>) -> PartitionCountFacts {
        let stored = files
            .iter()
            .filter_map(|file| file.partition_value.clone())
            .collect();
        PartitionCountFacts::new(true, false, stored, files)
    }

    fn resolve_at(
        requested: &[&str],
        snapshot: i64,
        at_head: bool,
        facts: &PartitionCountFacts,
        inlined: &[(i64, i64)],
    ) -> BTreeMap<String, PartitionRowCount> {
        let inlined = inlined
            .iter()
            .map(|&(data_file_id, row_id)| DuckLakeInlinedDelete {
                data_file_id,
                row_id,
            })
            .collect::<Vec<_>>();
        let read = CountRead {
            snapshot_id: snapshot,
            at_head,
            partition_id: SPEC,
            key_type: &DataType::Int32,
        };
        resolve_partition_counts(&values(requested), &read, Some(facts), &inlined)
    }

    fn resolve(
        requested: &[&str],
        snapshot: i64,
        files: Vec<PartitionCountFile>,
        inlined: &[(i64, i64)],
    ) -> BTreeMap<String, PartitionRowCount> {
        resolve_at(requested, snapshot, false, &facts(files), inlined)
    }

    fn deleted(file: PartitionCountFile, count: i64, max: Option<i64>) -> PartitionCountFile {
        file.with_delete_files(1, Some(count), max)
    }

    #[test]
    fn sums_files_and_subtracts_deletes() {
        let counts = resolve(
            &["1", "2", "3"],
            5,
            vec![
                file(1, "1", 3),
                deleted(file(2, "1", 10), 4, Some(5)),
                file(3, "2", 2),
                file(4, "9", 100),
            ],
            &[],
        );
        assert_eq!(counts["1"], PartitionRowCount::Exact(9));
        assert_eq!(counts["2"], PartitionRowCount::Exact(2));
        assert_eq!(counts["3"], PartitionRowCount::Exact(0));
        assert_eq!(counts.len(), 3);
    }

    #[test]
    fn inlined_deletes_count_once_and_only_in_range() {
        let counts = resolve(
            &["1"],
            5,
            vec![file(1, "1", 10)],
            &[(1, 0), (1, 0), (1, 3), (1, 99)],
        );
        assert_eq!(counts["1"], PartitionRowCount::Exact(8));
    }

    #[test]
    fn inlined_deletes_over_a_delete_file_are_unknown() {
        let counts = resolve(
            &["1", "2"],
            5,
            vec![deleted(file(1, "1", 10), 1, Some(5)), file(2, "2", 1)],
            &[(1, 4)],
        );
        assert_eq!(
            counts["1"],
            PartitionRowCount::Unknown(PartitionCountUnknown::InlinedDeletesWithDeleteFile)
        );
        assert_eq!(counts["2"], PartitionRowCount::Exact(1));
    }

    #[test]
    fn partial_files_are_exact_only_at_or_past_their_max() {
        let mut merged = file(1, "1", 10);
        merged.partial_max = Some(6);
        assert_eq!(
            resolve(&["1"], 5, vec![merged.clone()], &[])["1"],
            PartitionRowCount::Unknown(PartitionCountUnknown::PartialFile)
        );
        assert_eq!(
            resolve(&["1"], 6, vec![merged], &[])["1"],
            PartitionRowCount::Exact(10)
        );
    }

    #[test]
    fn a_delete_file_counts_by_its_partial_max_or_at_the_head() {
        let partial = deleted(file(1, "1", 10), 2, Some(9));
        assert_eq!(
            resolve(&["1"], 5, vec![partial.clone()], &[])["1"],
            PartitionRowCount::Unknown(PartitionCountUnknown::PartialDeleteFile)
        );
        assert_eq!(
            resolve(&["1"], 9, vec![partial], &[])["1"],
            PartitionRowCount::Exact(8)
        );

        // A delete file with no `partial_max` (a flush writes one) can begin
        // before deletions it holds: only the head is sure to have them all.
        let undated = facts(vec![deleted(file(1, "1", 10), 2, None)]);
        assert_eq!(
            resolve_at(&["1"], 5, false, &undated, &[])["1"],
            PartitionRowCount::Unknown(PartitionCountUnknown::PartialDeleteFile)
        );
        assert_eq!(
            resolve_at(&["1"], 5, true, &undated, &[])["1"],
            PartitionRowCount::Exact(8)
        );
    }

    #[test]
    fn delete_file_counts_are_checked() {
        let two = file(1, "1", 10).with_delete_files(2, None, None);
        assert_eq!(
            resolve(&["1"], 5, vec![two], &[])["1"],
            PartitionRowCount::Unknown(PartitionCountUnknown::MultipleDeleteFiles)
        );
        for count in [None, Some(-1), Some(11)] {
            let bad = file(1, "1", 10).with_delete_files(1, count, Some(5));
            assert_eq!(
                resolve(&["1"], 5, vec![bad], &[])["1"],
                PartitionRowCount::Unknown(PartitionCountUnknown::InvalidDeleteCount),
                "{count:?}"
            );
        }
    }

    #[test]
    fn missing_counts_are_unknown_not_zero() {
        let mut missing = file(1, "1", 0);
        missing.record_count = None;
        assert_eq!(
            resolve(&["1"], 5, vec![missing, file(2, "1", 3)], &[])["1"],
            PartitionRowCount::Unknown(PartitionCountUnknown::MissingRecordCount)
        );
    }

    #[test]
    fn an_unattributable_file_makes_every_value_unknown() {
        let mut other_spec = file(1, "9", 1);
        other_spec.partition_id = Some(SPEC - 1);
        let mut no_value = file(2, "9", 1);
        no_value.value_rows = 0;
        no_value.partition_value = None;
        for stray in [other_spec, no_value] {
            let counts = resolve(&["1"], 5, vec![file(3, "1", 1), stray], &[]);
            assert_eq!(
                counts["1"],
                PartitionRowCount::Unknown(PartitionCountUnknown::UnattributedFile)
            );
        }
    }

    #[test]
    fn spellings_must_be_canonical() {
        // A requested value no stored value could equal as text.
        let counts = resolve(&["02", "+2", "2"], 5, vec![file(1, "2", 4)], &[]);
        assert_eq!(
            counts["02"],
            PartitionRowCount::Unknown(PartitionCountUnknown::NonCanonicalValue)
        );
        assert_eq!(
            counts["+2"],
            PartitionRowCount::Unknown(PartitionCountUnknown::NonCanonicalValue)
        );
        assert_eq!(counts["2"], PartitionRowCount::Exact(4));
        // A stored value another writer spelled differently could hold rows of
        // a requested one.
        let mut stored = facts(vec![file(1, "2", 4)]);
        stored.stored_values.push("02".to_string());
        assert_eq!(
            resolve_at(&["2"], 5, false, &stored, &[])["2"],
            PartitionRowCount::Unknown(PartitionCountUnknown::UnattributedFile)
        );
        assert!(is_canonical_partition_value("it's a b", &DataType::Utf8));
        assert!(is_canonical_partition_value("-1", &DataType::Int32));
        assert!(!is_canonical_partition_value(" 1", &DataType::Int32));
    }

    #[test]
    fn an_overflowing_sum_is_unknown() {
        let counts = resolve(
            &["1"],
            5,
            vec![file(1, "1", i64::MAX), file(2, "1", i64::MAX), file(3, "1", i64::MAX)],
            &[],
        );
        assert_eq!(
            counts["1"],
            PartitionRowCount::Unknown(PartitionCountUnknown::Overflow)
        );
        let counts = resolve(
            &["1"],
            5,
            vec![file(1, "1", i64::MAX), file(2, "1", i64::MAX)],
            &[],
        );
        assert_eq!(counts["1"], PartitionRowCount::Exact(2 * i64::MAX as u64));
    }

    #[test]
    fn missing_snapshots_inlined_rows_and_unsupported_providers_are_unknown() {
        let requested = values(&["1"]);
        let read = CountRead {
            snapshot_id: 5,
            at_head: true,
            partition_id: SPEC,
            key_type: &DataType::Int32,
        };
        let inlined = PartitionCountFacts::new(true, true, Vec::new(), vec![file(1, "1", 1)]);
        assert_eq!(
            resolve_partition_counts(&requested, &read, Some(&inlined), &[])["1"],
            PartitionRowCount::Unknown(PartitionCountUnknown::InlinedRows)
        );
        let gone = PartitionCountFacts::new(false, false, Vec::new(), Vec::new());
        assert_eq!(
            resolve_partition_counts(&requested, &read, Some(&gone), &[])["1"],
            PartitionRowCount::Unknown(PartitionCountUnknown::SnapshotMissing)
        );
        assert_eq!(
            resolve_partition_counts(&requested, &read, None, &[])["1"],
            PartitionRowCount::Unknown(PartitionCountUnknown::Unsupported)
        );
    }

    #[test]
    fn a_file_listed_twice_counts_once() {
        let counts = resolve(&["1"], 5, vec![file(1, "1", 4), file(1, "1", 4)], &[]);
        assert_eq!(counts["1"], PartitionRowCount::Exact(4));
    }
}
