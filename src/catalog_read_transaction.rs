//! The read-only transaction behind a sqlx-backed provider's
//! [`ConsistentRead`](crate::metadata_provider::ConsistentRead), and the
//! connection every read of such a provider runs on.

use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering};

use sqlx::pool::PoolConnection;
use sqlx::{Database, Executor, Pool};
use tokio::sync::{MappedMutexGuard, Mutex, MutexGuard};

/// One pooled connection inside a read-only transaction, for the catalog reads
/// of one [`ConsistentRead`](crate::metadata_provider::ConsistentRead).
///
/// [`Self::commit`] ends the transaction and gives the connection back to the
/// pool. A provider bound to a committed read reads through the pool again.
/// Dropping the last handle to an open read rolls it back from a task on the
/// catalog runtime, and closes the connection instead of pooling it when the
/// rollback fails.
pub(crate) struct ReadTransaction<DB>
where
    DB: Database,
    for<'c> &'c mut DB::Connection: Executor<'c, Database = DB>,
{
    // A lock rather than a plain cell because DataFusion can plan several
    // scans at once, and a provider bound to the read can be shared by them:
    // their statements take turns on the connection.
    connection: Mutex<Option<PoolConnection<DB>>>,
    // Cleared when the read commits, so that a provider bound to it can tell
    // without waiting on the connection's lock.
    open: AtomicBool,
}

impl<DB> ReadTransaction<DB>
where
    DB: Database,
    for<'c> &'c mut DB::Connection: Executor<'c, Database = DB>,
{
    /// Take one connection from `pool` and run `begin` on it, statement by
    /// statement; the last one starts the transaction.
    pub(crate) async fn begin(
        pool: &Pool<DB>,
        begin: &[&'static str],
    ) -> std::result::Result<Self, sqlx::Error> {
        Self::begin_on(pool.acquire().await?, begin).await
    }

    /// Begin as [`Self::begin`] does, on a connection `pool` hands over
    /// without waiting for an idle one: an idle one, or a new one while the
    /// pool has fewer than its maximum. Losing the race for the pool's last
    /// free place can still make it wait briefly for a connection given back.
    ///
    /// `None` when every connection the pool may hold is taken, or when the
    /// connection cannot be opened or the transaction cannot begin on it. A
    /// read that is an optimisation, such as a pinned read, begins this way,
    /// so that a busy or failing pool degrades it rather than failing the
    /// query that wanted it. An idle connection is taken without the pool's
    /// checks before use, so one the server has closed fails to begin and is
    /// discarded.
    pub(crate) async fn try_begin(pool: &Pool<DB>, begin: &[&'static str]) -> Option<Self> {
        let connection = match pool.try_acquire() {
            Some(connection) => connection,
            // A connection the pool opens is waited for only as long as it
            // takes to open; a task that took the last free place first makes
            // this wait for a connection given back, as any acquire would.
            None if pool.size() < pool.options().get_max_connections() => {
                match pool.acquire().await {
                    Ok(connection) => connection,
                    Err(error) => {
                        tracing::debug!(%error, "no catalog connection to begin a read on");
                        return None;
                    },
                }
            },
            None => return None,
        };
        match Self::begin_on(connection, begin).await {
            Ok(read) => Some(read),
            Err(error) => {
                tracing::debug!(%error, "a catalog read failed to begin");
                None
            },
        }
    }

    /// Run `begin` on `connection`, statement by statement; the last one
    /// starts the transaction.
    async fn begin_on(
        mut connection: PoolConnection<DB>,
        begin: &[&'static str],
    ) -> std::result::Result<Self, sqlx::Error> {
        for statement in begin {
            if let Err(error) = sqlx::raw_sql(*statement).execute(&mut *connection).await {
                // The connection may be inside a transaction it cannot leave.
                connection.close_on_drop();
                return Err(error);
            }
        }
        Ok(Self {
            connection: Mutex::new(Some(connection)),
            open: AtomicBool::new(true),
        })
    }

    /// Whether the read has not been committed.
    pub(crate) fn is_open(&self) -> bool {
        self.open.load(Ordering::Acquire)
    }

    /// The read's connection while the read is open.
    pub(crate) async fn connection(&self) -> Option<MappedMutexGuard<'_, DB::Connection>> {
        MutexGuard::try_map(self.connection.lock().await, |connection| {
            connection.as_deref_mut()
        })
        .ok()
    }

    /// Commit the transaction and give the connection back to the pool.
    /// Committing a committed read does nothing.
    pub(crate) async fn commit(&self) -> std::result::Result<(), sqlx::Error> {
        self.end_with("COMMIT").await
    }

    /// Roll the transaction back and give the connection back to the pool:
    /// [`Self::commit`] for a read that wrote scratch state of its own, such
    /// as SQLite temporary tables, which must not outlive it on a pooled
    /// connection. Ending an ended read does nothing.
    #[cfg_attr(not(feature = "metadata-sqlite"), allow(dead_code))]
    pub(crate) async fn rollback(&self) -> std::result::Result<(), sqlx::Error> {
        self.end_with("ROLLBACK").await
    }

    async fn end_with(&self, statement: &'static str) -> std::result::Result<(), sqlx::Error> {
        let Some(mut connection) = self.connection.lock().await.take() else {
            return Ok(());
        };
        self.open.store(false, Ordering::Release);
        if let Err(error) = sqlx::raw_sql(statement).execute(&mut *connection).await {
            connection.close_on_drop();
            return Err(error);
        }
        Ok(())
    }
}

impl<DB> Drop for ReadTransaction<DB>
where
    DB: Database,
    for<'c> &'c mut DB::Connection: Executor<'c, Database = DB>,
{
    fn drop(&mut self) {
        // sqlx returns a connection to its pool from a task it spawns, which
        // needs a runtime. The catalog runtime always has one; the thread that
        // drops the last handle might not.
        if let Some(mut connection) = self.connection.get_mut().take() {
            crate::metadata_provider::catalog_runtime().spawn(async move {
                if sqlx::raw_sql("ROLLBACK")
                    .execute(&mut *connection)
                    .await
                    .is_err()
                {
                    connection.close_on_drop();
                }
            });
        }
    }
}

impl<DB> std::fmt::Debug for ReadTransaction<DB>
where
    DB: Database,
    for<'c> &'c mut DB::Connection: Executor<'c, Database = DB>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadTransaction").finish_non_exhaustive()
    }
}

/// The connection one catalog statement runs on: the open read's, or one from
/// the pool.
pub(crate) enum CatalogConnection<'a, DB: Database> {
    Pooled(PoolConnection<DB>),
    InRead(MappedMutexGuard<'a, DB::Connection>),
}

impl<DB: Database> CatalogConnection<'_, DB> {
    /// Whether the statement runs inside a consistent read's transaction.
    #[cfg_attr(not(feature = "metadata-postgres"), allow(dead_code))]
    pub(crate) fn in_read(&self) -> bool {
        matches!(self, Self::InRead(_))
    }
}

impl<DB: Database> Deref for CatalogConnection<'_, DB> {
    type Target = DB::Connection;

    fn deref(&self) -> &DB::Connection {
        match self {
            Self::Pooled(connection) => connection,
            Self::InRead(connection) => connection,
        }
    }
}

impl<DB: Database> DerefMut for CatalogConnection<'_, DB> {
    fn deref_mut(&mut self) -> &mut DB::Connection {
        match self {
            Self::Pooled(connection) => connection,
            Self::InRead(connection) => connection,
        }
    }
}

/// The connection for one catalog statement of a provider bound to `read`: the
/// read's while it is open, else one from `pool`.
///
/// Hold it for one statement, or for the statements of one helper, and never
/// across a call that takes another: inside a read the second call waits on
/// the first forever.
pub(crate) async fn catalog_connection<'a, DB>(
    read: Option<&'a ReadTransaction<DB>>,
    pool: &Pool<DB>,
) -> std::result::Result<CatalogConnection<'a, DB>, sqlx::Error>
where
    DB: Database,
    for<'c> &'c mut DB::Connection: Executor<'c, Database = DB>,
{
    if let Some(read) = read
        && let Some(connection) = read.connection().await
    {
        return Ok(CatalogConnection::InRead(connection));
    }
    Ok(CatalogConnection::Pooled(pool.acquire().await?))
}
