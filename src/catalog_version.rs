//! The layout version of a catalog this crate writes, and what a writer may do
//! with the catalog it opens.
//!
//! A catalog this crate creates records its layout version in
//! `ducklake_metadata` under [`LAYOUT_VERSION_KEY`] (or
//! [`MULTICATALOG_LAYOUT_VERSION_KEY`] for the PostgreSQL multicatalog layout).
//! Opening a writer reads those rows and official DuckDB's `version` row, and
//! nothing else: a current catalog runs no DDL. The rows are catalog-wide, and
//! the read names no other column, so it also works on a catalog from before
//! `ducklake_metadata` had a `scope` column.
//!
//! A catalog at an older layout version, or one an earlier release created
//! without recording a version, is refused until the caller runs the writer's
//! `migrate` function. Migration is never automatic, as in official DuckDB,
//! which migrates only when attached with `AUTOMATIC_MIGRATION`.
//!
//! A catalog at a newer layout version is written as it is when its minimum
//! writer version ([`MIN_WRITER_VERSION_KEY`], or
//! [`MULTICATALOG_MIN_WRITER_VERSION_KEY`]) is at most this release's layout
//! version, and refused otherwise. `migrate` leaves it alone.
//!
//! The crate's layout version is its own, not official's `version`, and the
//! crate writes and migrates only catalogs it created: a catalog official
//! DuckDB created (it has `version` and no crate layout version) is refused,
//! and nothing here changes it.

use crate::{DuckLakeError, Result};

/// The `ducklake_metadata` key holding the layout version of a single-catalog
/// layout this crate created.
pub const LAYOUT_VERSION_KEY: &str = "datafusion_ducklake_version";

/// The `ducklake_metadata` key holding the layout version of a PostgreSQL
/// multicatalog store.
pub const MULTICATALOG_LAYOUT_VERSION_KEY: &str = "datafusion_ducklake_multicatalog_version";

/// The single-catalog layout version this release writes.
pub const LAYOUT_VERSION: u64 = 1;

/// The multicatalog layout version this release writes.
pub const MULTICATALOG_LAYOUT_VERSION: u64 = 1;

/// The `ducklake_metadata` key holding the oldest single-catalog layout
/// version whose release may still write the catalog.
pub const MIN_WRITER_VERSION_KEY: &str = "datafusion_ducklake_min_writer_version";

/// The `ducklake_metadata` key holding the oldest multicatalog layout version
/// whose release may still write the store.
pub const MULTICATALOG_MIN_WRITER_VERSION_KEY: &str =
    "datafusion_ducklake_multicatalog_min_writer_version";

/// The oldest single-catalog layout version whose release may write a catalog
/// at [`LAYOUT_VERSION`]. Raise it to [`LAYOUT_VERSION`] only when a release
/// at an older version would write the new layout wrongly, as by leaving a new
/// required column unset. An addition older releases can ignore, such as an
/// index, raises [`LAYOUT_VERSION`] alone, so a rolling deploy or a rollback
/// across it keeps writing. It never goes down: a later release keeps the
/// minimum of every release before it, or raises it.
///
/// It guards writers only. Readers check no version, so a change older readers
/// would misread, one that changes what existing rows mean, needs more than
/// this.
pub const MIN_WRITER_VERSION: u64 = 1;

/// The oldest multicatalog layout version whose release may write a store at
/// [`MULTICATALOG_LAYOUT_VERSION`]; see [`MIN_WRITER_VERSION`].
pub const MULTICATALOG_MIN_WRITER_VERSION: u64 = 1;

/// A layout this crate writes, and the one a database might hold instead.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Layout {
    /// The `ducklake_metadata` key of this layout's version.
    pub(crate) key: &'static str,
    /// The version this release writes.
    pub(crate) current: u64,
    /// The `ducklake_metadata` key of this layout's minimum writer version.
    pub(crate) min_writer_key: &'static str,
    /// The minimum writer version this release records with [`Self::current`].
    pub(crate) min_writer: u64,
    /// The key of the other layout's version, which this layout refuses.
    pub(crate) other_key: &'static str,
    /// The name of the other layout, for the refusal.
    pub(crate) other_name: &'static str,
}

/// The single-catalog layouts (DuckDB, SQLite, MySQL, PostgreSQL).
pub(crate) const SINGLE_CATALOG: Layout = Layout {
    key: LAYOUT_VERSION_KEY,
    current: LAYOUT_VERSION,
    min_writer_key: MIN_WRITER_VERSION_KEY,
    min_writer: MIN_WRITER_VERSION,
    other_key: MULTICATALOG_LAYOUT_VERSION_KEY,
    other_name: "a PostgreSQL multicatalog store",
};

/// The PostgreSQL multicatalog layout.
#[cfg_attr(not(feature = "write-postgres"), allow(dead_code))]
pub(crate) const MULTICATALOG: Layout = Layout {
    key: MULTICATALOG_LAYOUT_VERSION_KEY,
    current: MULTICATALOG_LAYOUT_VERSION,
    min_writer_key: MULTICATALOG_MIN_WRITER_VERSION_KEY,
    min_writer: MULTICATALOG_MIN_WRITER_VERSION,
    other_key: LAYOUT_VERSION_KEY,
    other_name: "a single-catalog DuckLake catalog",
};

impl Layout {
    /// The `ducklake_metadata` keys a writer of this layout reads on open, as
    /// a SQL list.
    pub(crate) fn sql_keys(&self) -> String {
        format!(
            "('version', '{}', '{}', '{}')",
            self.key, self.min_writer_key, self.other_key
        )
    }

    /// The catalog-wide `ducklake_metadata` rows that record this release's
    /// layout, as `(key, value)`. A migration replaces both in the transaction
    /// that records them.
    pub(crate) fn stamp(&self) -> [(&'static str, String); 2] {
        [(self.key, self.current.to_string()), (self.min_writer_key, self.min_writer.to_string())]
    }
}

/// What a writer finds when it opens a database.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CatalogState {
    /// No catalog: no `ducklake_metadata` table. `initialize_schema` creates it.
    Absent,
    /// A catalog at the layout version this release writes.
    Current,
    /// A catalog this crate created that `migrate` must bring up to date: at
    /// this older layout version, or `None` when an earlier release created it
    /// without recording one.
    NeedsMigration(Option<u64>),
    /// A catalog a newer release wrote, at this layout version, whose minimum
    /// writer version admits this release. It is written as it is, and
    /// `migrate` leaves it alone.
    NewerCompatible(u64),
    /// A catalog a newer release wrote, at this layout version, that this
    /// release must not write.
    Newer(u64),
    /// A catalog official DuckDB created, at this DuckLake `version`.
    Official(String),
    /// A database holding the other layout.
    OtherLayout(&'static str),
}

/// Classify an existing catalog from its `ducklake_metadata` version rows,
/// `(key, value)` pairs read with [`Layout::sql_keys`]. A catalog with neither
/// version row reads as [`CatalogState::NeedsMigration`]`(None)`: an earlier
/// release created it, or a creation that stopped before recording its
/// version left it, and `migrate` brings it up to date either way. A catalog
/// at a newer version is [`CatalogState::NewerCompatible`] when its minimum
/// writer version is at most this release's version, and
/// [`CatalogState::Newer`] when it is above, or when the catalog records none.
pub(crate) fn classify(rows: &[(String, String)], layout: &Layout) -> Result<CatalogState> {
    let row = |key: &str| {
        rows.iter()
            .find(|(k, _)| k == key)
            .map(|(_, value)| value.as_str())
    };
    let parse = |key: &str, value: &str| {
        value.trim().parse::<u64>().map_err(|_| {
            DuckLakeError::InvalidConfig(format!("{key} is {value:?}, not a layout version"))
        })
    };
    if row(layout.other_key).is_some() {
        return Ok(CatalogState::OtherLayout(layout.other_name));
    }
    let Some(version) = row(layout.key) else {
        return Ok(match row("version") {
            Some(official) => CatalogState::Official(official.to_string()),
            None => CatalogState::NeedsMigration(None),
        });
    };
    let version = parse(layout.key, version)?;
    Ok(match version.cmp(&layout.current) {
        std::cmp::Ordering::Equal => CatalogState::Current,
        std::cmp::Ordering::Less => CatalogState::NeedsMigration(Some(version)),
        std::cmp::Ordering::Greater => {
            let min_writer = match row(layout.min_writer_key) {
                Some(min_writer) => parse(layout.min_writer_key, min_writer)?,
                None => version,
            };
            if min_writer <= layout.current {
                CatalogState::NewerCompatible(version)
            } else {
                CatalogState::Newer(version)
            }
        },
    })
}

/// Refuse to open a catalog a writer cannot write as it is. `migrate` names the
/// function that migrates it.
pub(crate) fn require_writable(state: &CatalogState, migrate: &str) -> Result<()> {
    match state {
        CatalogState::Absent | CatalogState::Current | CatalogState::NewerCompatible(_) => Ok(()),
        CatalogState::NeedsMigration(version) => {
            let found = match version {
                None => "this catalog records no datafusion-ducklake layout version: an earlier \
                         release created it, or its creation stopped before recording one"
                    .to_string(),
                Some(version) => format!("this catalog is at layout version {version}"),
            };
            Err(DuckLakeError::MigrationRequired(format!(
                "{found}; call {migrate} to migrate it"
            )))
        },
        refused => Err(refusal(refused)),
    }
}

/// Refuse to migrate what this crate does not migrate: a catalog a newer
/// release wrote, one official DuckDB created, or the other layout. `migrate`
/// leaves a [`CatalogState::NewerCompatible`] catalog alone before asking; it
/// is refused here too, so no migration records an older version over it.
pub(crate) fn require_migratable(state: &CatalogState) -> Result<()> {
    match state {
        CatalogState::NewerCompatible(_)
        | CatalogState::Newer(_)
        | CatalogState::Official(_)
        | CatalogState::OtherLayout(_) => Err(refusal(state)),
        _ => Ok(()),
    }
}

fn refusal(state: &CatalogState) -> DuckLakeError {
    DuckLakeError::Unsupported(match state {
        CatalogState::Newer(version) => format!(
            "this catalog is at layout version {version}, newer than this release writes; \
             upgrade datafusion-ducklake"
        ),
        CatalogState::NewerCompatible(version) => format!(
            "this catalog is at layout version {version}, newer than this release migrates to"
        ),
        CatalogState::Official(version) => format!(
            "official DuckDB created this catalog (DuckLake {version}); datafusion-ducklake \
             writes and migrates only catalogs it created"
        ),
        CatalogState::OtherLayout(name) => format!("this database holds {name}"),
        other => format!("unexpected catalog state {other:?}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn classifies_by_the_version_rows() {
        let key = LAYOUT_VERSION_KEY;
        let at = |current| Layout {
            current,
            ..SINGLE_CATALOG
        };
        assert_eq!(
            classify(&rows(&[(key, "1")]), &at(1)).unwrap(),
            CatalogState::Current
        );
        assert_eq!(
            classify(&rows(&[(key, "1")]), &at(2)).unwrap(),
            CatalogState::NeedsMigration(Some(1))
        );
        assert_eq!(
            classify(&rows(&[(key, "3")]), &at(2)).unwrap(),
            CatalogState::Newer(3)
        );
        assert_eq!(
            classify(&[], &at(1)).unwrap(),
            CatalogState::NeedsMigration(None)
        );
        assert_eq!(
            classify(&rows(&[("version", "1.0")]), &at(1)).unwrap(),
            CatalogState::Official("1.0".to_string())
        );
        assert_eq!(
            classify(&rows(&[(MULTICATALOG_LAYOUT_VERSION_KEY, "1")]), &at(1)).unwrap(),
            CatalogState::OtherLayout(SINGLE_CATALOG.other_name)
        );
        assert_eq!(
            classify(&rows(&[(LAYOUT_VERSION_KEY, "1")]), &MULTICATALOG).unwrap(),
            CatalogState::OtherLayout(MULTICATALOG.other_name)
        );
        assert!(classify(&rows(&[(key, "one")]), &at(1)).is_err());
    }

    /// A catalog a newer release stamped is writable when its minimum writer
    /// version is at most this release's, and refused otherwise or when it
    /// records none.
    #[test]
    fn a_newer_catalog_is_writable_down_to_its_minimum_writer_version() {
        let key = LAYOUT_VERSION_KEY;
        let min = MIN_WRITER_VERSION_KEY;
        let at = |current| Layout {
            current,
            ..SINGLE_CATALOG
        };
        assert_eq!(
            classify(&rows(&[(key, "3"), (min, "1")]), &at(2)).unwrap(),
            CatalogState::NewerCompatible(3)
        );
        assert_eq!(
            classify(&rows(&[(key, "3"), (min, "2")]), &at(2)).unwrap(),
            CatalogState::NewerCompatible(3)
        );
        assert_eq!(
            classify(&rows(&[(key, "3"), (min, "3")]), &at(2)).unwrap(),
            CatalogState::Newer(3)
        );
        assert_eq!(
            classify(&rows(&[(key, "3")]), &at(2)).unwrap(),
            CatalogState::Newer(3)
        );
        // At or below this release's version, the minimum changes nothing.
        assert_eq!(
            classify(&rows(&[(key, "2"), (min, "1")]), &at(2)).unwrap(),
            CatalogState::Current
        );
        assert_eq!(
            classify(&rows(&[(key, "1"), (min, "1")]), &at(2)).unwrap(),
            CatalogState::NeedsMigration(Some(1))
        );
        assert_eq!(
            classify(
                &rows(&[
                    (MULTICATALOG_LAYOUT_VERSION_KEY, "3"),
                    (MULTICATALOG_MIN_WRITER_VERSION_KEY, "1")
                ]),
                &Layout {
                    current: 2,
                    ..MULTICATALOG
                }
            )
            .unwrap(),
            CatalogState::NewerCompatible(3)
        );
        // The other layout's minimum is not this layout's.
        assert_eq!(
            classify(
                &rows(&[(key, "3"), (MULTICATALOG_MIN_WRITER_VERSION_KEY, "1")]),
                &at(2)
            )
            .unwrap(),
            CatalogState::Newer(3)
        );
        assert!(classify(&rows(&[(key, "3"), (min, "one")]), &at(2)).is_err());
    }

    #[test]
    fn the_stamp_records_the_version_and_the_minimum_writer_version() {
        assert_eq!(
            SINGLE_CATALOG.stamp(),
            [
                (LAYOUT_VERSION_KEY, LAYOUT_VERSION.to_string()),
                (MIN_WRITER_VERSION_KEY, MIN_WRITER_VERSION.to_string()),
            ]
        );
        assert_eq!(
            MULTICATALOG.stamp(),
            [
                (
                    MULTICATALOG_LAYOUT_VERSION_KEY,
                    MULTICATALOG_LAYOUT_VERSION.to_string()
                ),
                (
                    MULTICATALOG_MIN_WRITER_VERSION_KEY,
                    MULTICATALOG_MIN_WRITER_VERSION.to_string()
                ),
            ]
        );
        const { assert!(MIN_WRITER_VERSION <= LAYOUT_VERSION) };
        const { assert!(MULTICATALOG_MIN_WRITER_VERSION <= MULTICATALOG_LAYOUT_VERSION) };
    }

    #[test]
    fn writers_refuse_all_but_a_current_or_new_catalog() {
        assert!(require_writable(&CatalogState::Absent, "m").is_ok());
        assert!(require_writable(&CatalogState::Current, "m").is_ok());
        assert!(require_writable(&CatalogState::NewerCompatible(9), "m").is_ok());
        let refused = require_writable(&CatalogState::NeedsMigration(None), "X::migrate");
        assert!(
            matches!(&refused, Err(DuckLakeError::MigrationRequired(m)) if m.contains("X::migrate")),
            "{refused:?}"
        );
        for state in [
            CatalogState::Newer(9),
            CatalogState::Official("1.0".to_string()),
            CatalogState::OtherLayout("x"),
        ] {
            assert!(
                matches!(
                    require_writable(&state, "m"),
                    Err(DuckLakeError::Unsupported(_))
                ),
                "{state:?}"
            );
        }
    }

    #[test]
    fn migration_refuses_what_this_crate_did_not_create() {
        assert!(require_migratable(&CatalogState::NeedsMigration(None)).is_ok());
        assert!(require_migratable(&CatalogState::NeedsMigration(Some(1))).is_ok());
        assert!(require_migratable(&CatalogState::Newer(2)).is_err());
        assert!(require_migratable(&CatalogState::NewerCompatible(2)).is_err());
        assert!(require_migratable(&CatalogState::Official("1.0".to_string())).is_err());
        assert!(require_migratable(&CatalogState::OtherLayout("x")).is_err());
    }
}
