//! The read-only transaction behind a sqlx-backed provider's
//! [`ConsistentRead`](crate::metadata_provider::ConsistentRead), and the
//! connection every read of such a provider runs on.

use std::ops::{Deref, DerefMut};

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
        let mut connection = pool.acquire().await?;
        for statement in begin {
            if let Err(error) = sqlx::raw_sql(*statement).execute(&mut *connection).await {
                // The connection may be inside a transaction it cannot leave.
                connection.close_on_drop();
                return Err(error);
            }
        }
        Ok(Self {
            connection: Mutex::new(Some(connection)),
        })
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
        let Some(mut connection) = self.connection.lock().await.take() else {
            return Ok(());
        };
        if let Err(error) = sqlx::raw_sql("COMMIT").execute(&mut *connection).await {
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

/// Run `read`, a statement whose failure its caller tolerates, on
/// `connection`; under a savepoint when `savepoint` is set, which a statement
/// inside a consistent read's transaction needs.
///
/// PostgreSQL ends a transaction at its first failed statement, and every
/// later statement of the transaction fails until it ends. A savepoint undoes
/// the failure, so the read goes on as it does outside a transaction. It costs
/// two round trips, so a provider sets it only around a statement that can fail
/// on a current catalog.
#[cfg(feature = "metadata-postgres")]
pub(crate) async fn tolerating_failure<T>(
    connection: &mut sqlx::PgConnection,
    savepoint: bool,
    read: impl AsyncFnOnce(&mut sqlx::PgConnection) -> std::result::Result<T, sqlx::Error>,
) -> std::result::Result<T, sqlx::Error> {
    if !savepoint {
        return read(connection).await;
    }
    sqlx::raw_sql("SAVEPOINT ducklake_tolerated_read")
        .execute(&mut *connection)
        .await?;
    match read(&mut *connection).await {
        Ok(value) => {
            sqlx::raw_sql("RELEASE SAVEPOINT ducklake_tolerated_read")
                .execute(&mut *connection)
                .await?;
            Ok(value)
        },
        Err(error) => {
            sqlx::raw_sql(
                "ROLLBACK TO SAVEPOINT ducklake_tolerated_read; \
                 RELEASE SAVEPOINT ducklake_tolerated_read",
            )
            .execute(&mut *connection)
            .await?;
            Err(error)
        },
    }
}
