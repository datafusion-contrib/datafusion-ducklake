//! DuckLake snapshot-collision classification and bounded metadata retries.

use std::future::Future;
use std::time::Duration;

use crate::Result;
use crate::error::DuckLakeError;
use crate::metadata_writer::SnapshotChanges;

/// Default maximum metadata commit retries after the initial attempt.
pub const DEFAULT_MAX_RETRY_COUNT: u32 = 10;
/// Default multiplier applied to each successive retry delay.
pub const DEFAULT_RETRY_BACKOFF: f64 = 1.5;
/// Default delay before the first retry, in milliseconds.
pub const DEFAULT_RETRY_WAIT_MS: u64 = 100;

/// Bounded retry settings for optimistic DuckLake metadata commits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConflictRetryConfig {
    /// Maximum retries after the initial metadata commit attempt.
    pub max_count: u32,
    /// Exponential multiplier for successive retry delays.
    pub backoff: f64,
    /// Delay before the first retry, in milliseconds.
    pub wait_ms: u64,
}

impl Default for ConflictRetryConfig {
    fn default() -> Self {
        Self {
            max_count: DEFAULT_MAX_RETRY_COUNT,
            backoff: DEFAULT_RETRY_BACKOFF,
            wait_ms: DEFAULT_RETRY_WAIT_MS,
        }
    }
}

impl ConflictRetryConfig {
    pub(crate) fn validate(self) -> Result<()> {
        if !self.backoff.is_finite() || self.backoff <= 0.0 {
            return Err(DuckLakeError::InvalidConfig(format!(
                "conflict retry backoff must be finite and positive, got {}",
                self.backoff
            )));
        }
        Ok(())
    }

    fn delay(self, retry: u32) -> Duration {
        let delay_ms =
            (self.wait_ms as f64 * self.backoff.powf(f64::from(retry))).min(u64::MAX as f64) as u64;
        Duration::from_millis(delay_ms)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommitChange {
    Insert,
    Delete,
    Replace,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CommitTarget<'a> {
    pub change: CommitChange,
    pub schema_id: i64,
    pub schema_name: &'a str,
    pub table_id: i64,
    pub table_name: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SnapshotChange {
    CreatedSchema(String),
    CreatedTable(String),
    CreatedView(String),
    Inserted(i64),
    Deleted(i64),
    Compacted(i64),
    DroppedSchema(i64),
    DroppedTable(i64),
    DroppedView,
    AlteredTable(i64),
    AlteredView,
}

/// Retry one metadata commit after optimistic snapshot-id collisions.
///
/// Data-file creation and upload happen before this function is entered. Only
/// `commit` repeats, with a new `(snapshot_id, base_snapshot)` pair derived from
/// the intervening committed snapshots.
pub(crate) async fn commit_with_retry<T, C, S, SF>(
    config: ConflictRetryConfig,
    initial_snapshot: i64,
    initial_base: i64,
    targets: &[CommitTarget<'_>],
    mut commit: C,
    mut snapshots: S,
) -> Result<T>
where
    C: FnMut(i64, i64) -> Result<T>,
    S: FnMut(i64) -> SF,
    SF: Future<Output = Result<Vec<SnapshotChanges>>>,
{
    config.validate()?;
    let mut snapshot_id = initial_snapshot;
    let mut base_snapshot = initial_base;
    let mut retry_count = 0;

    loop {
        match commit(snapshot_id, base_snapshot) {
            Ok(result) => return Ok(result),
            Err(DuckLakeError::SnapshotCollision {
                snapshot_id: collided_snapshot,
                message,
            }) => {
                if retry_count >= config.max_count {
                    return Err(DuckLakeError::Conflict(format!(
                        "exceeded the maximum snapshot commit retry count of {} after collision \
                         at snapshot {collided_snapshot}: {message}",
                        config.max_count
                    )));
                }

                let intervening = snapshots(base_snapshot).await?;
                let new_base = classify_intervening(base_snapshot, &intervening, targets)?;
                snapshot_id = new_base.checked_add(1).ok_or_else(|| {
                    DuckLakeError::Conflict(
                        "cannot retry after the maximum snapshot identifier".to_string(),
                    )
                })?;
                base_snapshot = new_base;

                let delay = config.delay(retry_count);
                retry_count += 1;
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
            },
            Err(e) => return Err(e),
        }
    }
}

fn classify_intervening(
    base_snapshot: i64,
    snapshots: &[SnapshotChanges],
    targets: &[CommitTarget<'_>],
) -> Result<i64> {
    let Some(last) = snapshots.last() else {
        return Err(DuckLakeError::Conflict(format!(
            "snapshot commit collided after base {base_snapshot}, but no intervening snapshot \
             was available for conflict resolution"
        )));
    };
    let mut previous = base_snapshot;
    for snapshot in snapshots {
        if snapshot.snapshot_id <= previous {
            return Err(DuckLakeError::Conflict(format!(
                "snapshot conflict history is not strictly increasing after {previous}: got {}",
                snapshot.snapshot_id
            )));
        }
        previous = snapshot.snapshot_id;
        let changes_made = snapshot.changes_made.as_deref().ok_or_else(|| {
            DuckLakeError::Conflict(format!(
                "snapshot {} has no changes_made value, so its collision cannot be retried safely",
                snapshot.snapshot_id
            ))
        })?;
        let changes = parse_changes(changes_made).ok_or_else(|| {
            DuckLakeError::Conflict(format!(
                "snapshot {} has an unrecognized changes_made value: {changes_made}",
                snapshot.snapshot_id
            ))
        })?;
        if let Some((target, change)) = targets.iter().find_map(|target| {
            changes
                .iter()
                .find(|change| conflicts(*target, change))
                .map(|change| (target, change))
        }) {
            return Err(DuckLakeError::Conflict(format!(
                "snapshot {} change {change:?} conflicts with {:?} on table {}",
                snapshot.snapshot_id, target.change, target.table_id
            )));
        }
    }
    Ok(last.snapshot_id)
}

fn conflicts(target: CommitTarget<'_>, change: &SnapshotChange) -> bool {
    match change {
        SnapshotChange::CreatedSchema(name) => name == target.schema_name,
        SnapshotChange::DroppedSchema(schema_id) => {
            target.schema_id < 0 || *schema_id == target.schema_id
        },
        SnapshotChange::CreatedTable(name) | SnapshotChange::CreatedView(name) => {
            table_name(name) == target.table_name
        },
        SnapshotChange::DroppedTable(table_id) | SnapshotChange::AlteredTable(table_id) => {
            *table_id == target.table_id
        },
        SnapshotChange::Inserted(table_id) => {
            *table_id == target.table_id && target.change == CommitChange::Replace
        },
        SnapshotChange::Deleted(table_id) => {
            *table_id == target.table_id
                && matches!(target.change, CommitChange::Delete | CommitChange::Replace)
        },
        SnapshotChange::Compacted(table_id) => {
            *table_id == target.table_id
                && matches!(target.change, CommitChange::Delete | CommitChange::Replace)
        },
        SnapshotChange::DroppedView | SnapshotChange::AlteredView => false,
    }
}

fn table_name(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or(name).trim_matches('"')
}

fn parse_changes(changes_made: &str) -> Option<Vec<SnapshotChange>> {
    split_changes(changes_made)
        .into_iter()
        .filter(|change| !change.trim().is_empty())
        .map(parse_change)
        .collect()
}

fn parse_change(change: &str) -> Option<SnapshotChange> {
    let (kind, value) = change.trim().split_once(':')?;
    let value = value.trim();
    Some(match kind {
        "created_schema" => SnapshotChange::CreatedSchema(unquote(value)?),
        "created_table" => SnapshotChange::CreatedTable(unquote(value)?),
        "created_view" => SnapshotChange::CreatedView(unquote(value)?),
        "inserted_into_table" => SnapshotChange::Inserted(value.parse().ok()?),
        "deleted_from_table" => SnapshotChange::Deleted(value.parse().ok()?),
        "compacted_table" => SnapshotChange::Compacted(value.parse().ok()?),
        "dropped_schema" => SnapshotChange::DroppedSchema(value.parse().ok()?),
        "dropped_table" => SnapshotChange::DroppedTable(value.parse().ok()?),
        "dropped_view" => SnapshotChange::DroppedView,
        "altered_table" => SnapshotChange::AlteredTable(value.parse().ok()?),
        "altered_view" => SnapshotChange::AlteredView,
        _ => return None,
    })
}

fn split_changes(changes_made: &str) -> Vec<&str> {
    let mut changes = Vec::new();
    let mut start = 0;
    let mut quoted = false;
    let mut chars = changes_made.char_indices().peekable();
    while let Some((index, character)) = chars.next() {
        match character {
            '"' if quoted && chars.peek().is_some_and(|(_, next)| *next == '"') => {
                chars.next();
            },
            '"' => quoted = !quoted,
            ',' if !quoted => {
                changes.push(&changes_made[start..index]);
                start = index + 1;
            },
            _ => {},
        }
    }
    changes.push(&changes_made[start..]);
    changes
}

fn unquote(value: &str) -> Option<String> {
    if let Some(inner) = value.strip_prefix('"').and_then(|v| v.strip_suffix('"')) {
        Some(inner.replace("\"\"", "\""))
    } else if value.contains('"') {
        None
    } else {
        Some(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    fn insert_target() -> CommitTarget<'static> {
        CommitTarget {
            change: CommitChange::Insert,
            schema_id: 4,
            schema_name: "main",
            table_id: 7,
            table_name: "events",
        }
    }

    fn collision(snapshot_id: i64) -> DuckLakeError {
        DuckLakeError::SnapshotCollision {
            snapshot_id,
            message: "primary key violation".to_string(),
        }
    }

    #[test]
    fn defaults_match_ducklake_extension_settings() {
        assert_eq!(
            ConflictRetryConfig::default(),
            ConflictRetryConfig {
                max_count: 10,
                backoff: 1.5,
                wait_ms: 100,
            }
        );
    }

    #[test]
    fn parses_quoted_names_with_commas_and_escaped_quotes() {
        assert_eq!(
            parse_changes(
                "created_schema:\"a,b\",created_table:\"event \"\"log\"\"\",inserted_into_table:7"
            ),
            Some(vec![
                SnapshotChange::CreatedSchema("a,b".to_string()),
                SnapshotChange::CreatedTable("event \"log\"".to_string()),
                SnapshotChange::Inserted(7),
            ])
        );
    }

    #[test]
    fn applies_ducklake_table_conflict_rules() {
        let insert = insert_target();
        assert!(!conflicts(insert, &SnapshotChange::Inserted(7)));
        assert!(!conflicts(insert, &SnapshotChange::Deleted(7)));
        assert!(conflicts(
            insert,
            &SnapshotChange::CreatedSchema("main".to_string())
        ));
        assert!(conflicts(
            insert,
            &SnapshotChange::CreatedTable("events".to_string())
        ));
        assert!(conflicts(insert, &SnapshotChange::AlteredTable(7)));
        assert!(conflicts(insert, &SnapshotChange::DroppedTable(7)));

        let delete = CommitTarget {
            change: CommitChange::Delete,
            ..insert
        };
        assert!(!conflicts(delete, &SnapshotChange::Inserted(7)));
        assert!(conflicts(delete, &SnapshotChange::Deleted(7)));
        assert!(conflicts(delete, &SnapshotChange::Compacted(7)));
        assert!(!conflicts(delete, &SnapshotChange::Deleted(8)));

        let delete_without_schema_id = CommitTarget {
            schema_id: -1,
            ..delete
        };
        assert!(conflicts(
            delete_without_schema_id,
            &SnapshotChange::DroppedSchema(99)
        ));
    }

    #[tokio::test]
    async fn retries_commuting_insert_metadata_only() {
        let attempts = RefCell::new(Vec::new());
        let result = commit_with_retry(
            ConflictRetryConfig {
                max_count: 2,
                backoff: 1.0,
                wait_ms: 0,
            },
            2,
            1,
            &[insert_target()],
            |snapshot_id, base_snapshot| {
                attempts.borrow_mut().push((snapshot_id, base_snapshot));
                if base_snapshot == 1 {
                    Err(collision(snapshot_id))
                } else {
                    Ok(snapshot_id)
                }
            },
            |base_snapshot| async move {
                Ok(vec![SnapshotChanges {
                    snapshot_id: base_snapshot + 1,
                    changes_made: Some("inserted_into_table:7".to_string()),
                }])
            },
        )
        .await
        .unwrap();

        assert_eq!(result, 3);
        assert_eq!(*attempts.borrow(), vec![(2, 1), (3, 2)]);
    }

    #[tokio::test]
    async fn logical_conflict_aborts_without_second_commit() {
        let attempts = RefCell::new(0);
        let error = commit_with_retry(
            ConflictRetryConfig {
                max_count: 2,
                backoff: 1.0,
                wait_ms: 0,
            },
            2,
            1,
            &[insert_target()],
            |snapshot_id, _| {
                *attempts.borrow_mut() += 1;
                Err::<i64, _>(collision(snapshot_id))
            },
            |_| async {
                Ok(vec![SnapshotChanges {
                    snapshot_id: 2,
                    changes_made: Some("altered_table:7".to_string()),
                }])
            },
        )
        .await
        .unwrap_err();

        assert!(matches!(error, DuckLakeError::Conflict(_)));
        assert!(error.to_string().contains("AlteredTable(7)"));
        assert_eq!(*attempts.borrow(), 1);
    }

    #[tokio::test]
    async fn retry_exhaustion_reports_bound_and_attempt_count() {
        let attempts = RefCell::new(0);
        let error = commit_with_retry(
            ConflictRetryConfig {
                max_count: 2,
                backoff: 1.0,
                wait_ms: 0,
            },
            2,
            1,
            &[insert_target()],
            |snapshot_id, _| {
                *attempts.borrow_mut() += 1;
                Err::<i64, _>(collision(snapshot_id))
            },
            |base_snapshot| async move {
                Ok(vec![SnapshotChanges {
                    snapshot_id: base_snapshot + 1,
                    changes_made: Some("inserted_into_table:7".to_string()),
                }])
            },
        )
        .await
        .unwrap_err();

        assert_eq!(*attempts.borrow(), 3);
        assert!(matches!(error, DuckLakeError::Conflict(_)));
        assert!(
            error
                .to_string()
                .contains("maximum snapshot commit retry count of 2")
        );
    }

    #[tokio::test]
    async fn invalid_backoff_fails_before_the_first_commit() {
        let attempts = RefCell::new(0);
        let error = commit_with_retry(
            ConflictRetryConfig {
                max_count: 2,
                backoff: f64::NAN,
                wait_ms: 0,
            },
            2,
            1,
            &[insert_target()],
            |_, _| {
                *attempts.borrow_mut() += 1;
                Ok(2)
            },
            |_| async { Ok(Vec::new()) },
        )
        .await
        .unwrap_err();

        assert!(matches!(error, DuckLakeError::InvalidConfig(_)));
        assert_eq!(*attempts.borrow(), 0);
    }
}
