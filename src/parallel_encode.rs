//! Encoding a data file's row groups on several threads.
//!
//! A [`RowGroupQueue`] cuts the batches of a write into row groups and encodes
//! each one as a job of its own on an [`EncodePool`], at most one job per pool
//! thread at a time. Finished row groups come back in the order their rows
//! were written, whichever job finished first, and the caller appends them to
//! its file with a [`RowGroupAppender`], or to the next file once one is full.
//! Row groups share no state: each job has its own column writers, so the
//! heaviest column of a file is encoded on as many threads as there are row
//! groups in flight.
//!
//! Official DuckLake's writer, DuckDB's Parquet `COPY`, also prepares row
//! groups on several threads (`PrepareRowGroup`) and writes them to the file
//! one at a time (`FlushRowGroup`). An insert into a DuckLake table does not
//! preserve insertion order, so DuckDB copies it in parallel
//! (`PARALLEL_COPY_TO_FILE`) and appends each row group as it is ready, in
//! whatever order its threads finish them. Rows kept in write order are one
//! of the orders that copy can produce.
//!
//! # Row groups
//!
//! A row group closes when it holds the row cap's rows
//! (`WriterProperties::max_row_group_row_count`), exactly, or when its input
//! reaches the byte cap (`WriterProperties::max_row_group_bytes`), counted as
//! the logical size of its rows in memory
//! ([`crate::write_encoding::sliced_memory_size`]), as DuckDB counts its
//! buffered input against `row_group_size_bytes`. A batch is split where a row
//! group closes, at the batch's average row size; a row larger than the byte
//! cap is a row group of its own. `ArrowWriter` instead closes a row group by
//! the running estimate of its encoded size, which only a sequential writer
//! has, so a byte cap gives row groups of other sizes here. The row cap alone
//! gives the same ones.
//!
//! Each row group's encoding is chosen from its own rows
//! ([`crate::write_encoding::props_for_row_group`]), as DuckDB chooses each
//! column chunk's; a file written on one thread takes its first row group's
//! choice for the whole file. The values and their order are the same either
//! way; the bytes of the file are not, and so neither are its column sizes.
//!
//! Each job also folds its rows into a [`WrittenLeafStats`], which the caller
//! merges in row-group order ([`WrittenLeafStats::merge_ordered`]), so no pass
//! over the values is left on the calling thread. The merge is one pass over
//! the values but for string and blob bounds cut short of a value's 256th
//! byte, which it can leave looser, or drop; see its docs.
//!
//! # Memory
//!
//! A pool is either a write's own ([`EncodePool::new`], from the write's
//! `encode_parallelism`) or one a caller shares across all its writes
//! ([`EncodePool::try_new`]). Both count the row groups out, handed to the
//! pool and not yet appended, across every queue they serve, and charge each
//! row group out to the pool's byte budget: twice its rows' size in memory
//! (its input, and as much again for the pages it encodes) until its job
//! finishes, then what its column chunks hold, until it is appended. They
//! differ in how a row group is admitted.
//!
//! A write's own pool admits loosely. The rows of a row group are charged
//! while it is assembled. When a row group is ready and the budget is spent,
//! or the pool already has `2 × threads` row groups out, its queue appends
//! its own earliest one before handing out another. A queue with nothing out
//! hands it out anyway, so a row group larger than the budget is still
//! written, and no queue waits on another's. A write therefore has at most
//! `2 × threads` row groups out, plus one for each other queue holding rows:
//! each file sequence of an unpartitioned write or lane, and each open
//! partition file (up to `max_open_partitions`) of a partitioned one.
//!
//! A shared pool admits strictly ([`RowGroupQueue::dispatch_checked`]). A row
//! group is handed out only when its charge fits in what the budget has left
//! and the pool has fewer than `2 × threads` row groups out, both checked and
//! taken in one atomic step each. When it does not fit, its queue appends its
//! own earliest row group out, which gives that charge back, and tries
//! again; once the queue has nothing out it encodes the row group on its own
//! thread, as a one-thread write does, and appends it at once. A write is
//! never refused for want of budget, and no queue waits on another's row
//! groups. A row group's charge is taken before it is handed out and is never
//! grown past the budget: a job whose chunks hold more than it was charged
//! keeps the charge it had unless the budget has room for the rest. So the
//! charges of the row groups out, across every write sharing the pool, never
//! pass the budget, and at most `2 × threads` are out; beyond that each queue
//! holds the one row group it is assembling, or encoding on its own thread,
//! as a one-thread writer holds one. The rows being assembled are not
//! charged.
//!
//! # Failure
//!
//! A row group that fails to encode, or to append, fails its queue for good:
//! every later push and finish of it returns an error, so a file missing the
//! row group's rows is never finished.
//!
//! Not counted: buffers a batch slice keeps alive beyond its own rows (a slice
//! of a larger batch pins the whole parent), the encoders' working memory
//! beyond the pages they hold, the distinct values counted to choose a row
//! group's encoding, the bloom filters and page indexes a file keeps until
//! its footer is written, and, in a shared pool, the row group each queue
//! assembles or encodes on its own thread and the pages of a job beyond its
//! charge that the budget has no room for. The budget bounds what is queued,
//! not the process's memory.
//!
//! # What runs on one thread instead
//!
//! Content-defined chunking (`WriterProperties::content_defined_chunking`)
//! cuts pages across a whole column, and an encrypted file binds each column
//! chunk to its file and row-group position; [`supports`] refuses both, and
//! the caller writes those files with `ArrowWriter`.

use std::collections::BTreeMap;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, MutexGuard};

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_writer::{ArrowColumnChunk, ArrowColumnWriter, compute_leaves};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;

use crate::Result;
use crate::error::DuckLakeError;
use crate::stats_collect::WrittenLeafStats;
use crate::write_encoding::{DataFileProperties, props_for_row_group, sliced_memory_size};

/// Whether a file written with `props` can have its row groups encoded on a
/// pool. See the module docs.
pub(crate) fn supports(props: &WriterProperties) -> bool {
    #[cfg(feature = "encryption")]
    if props.file_encryption_properties().is_some() {
        return false;
    }
    props.content_defined_chunking().is_none()
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

type Job = Box<dyn FnOnce() + Send>;

/// Threads that encode the row groups of Parquet data files, with a byte
/// budget and a count of the row groups handed to them.
///
/// Without one, a write that encodes on several threads
/// ([`DuckLakeWriteOptions::encode_parallelism`](crate::DuckLakeWriteOptions::encode_parallelism))
/// makes a pool of its own. [`EncodePool::try_new`] makes one a caller shares
/// across all its writes
/// ([`DuckLakeWriteOptions::encode_pool`](crate::DuckLakeWriteOptions::encode_pool),
/// [`DuckLakeTableWriter::with_encode_pool`](crate::DuckLakeTableWriter::with_encode_pool)),
/// as DuckDB shares one thread pool and one memory limit across the
/// connections of a database instance: its threads and its budget are then
/// limits on every write together, not on each.
///
/// A row group goes to a shared pool only when it fits: its charge, twice
/// its rows' size in memory (its input, and as much again for the pages it
/// encodes), within what the budget has left, and fewer than `2 × threads`
/// row groups out (handed to the pool and not yet appended to their files)
/// across every write. Once encoded, its charge becomes what its column
/// chunks hold, grown only within the budget, until it is appended. A row
/// group that does not fit is not refused and does not wait for another
/// write: its write first appends its own row groups out, which gives their
/// charge back, and once it has none out, encodes the row group on its own
/// thread, as a single-threaded write does, and appends it.
///
/// So the charges of the row groups out never add up to more than the
/// budget, and at most `2 × threads` are out, whatever the number of writes.
/// Beyond that, each file a write is filling (each file sequence of a write
/// or of one of its lanes, and each open partition file of a partitioned
/// write) holds the one row group it is assembling or encoding on its own
/// thread, as a single-threaded writer holds one.
///
/// This is not a bound on the process's memory. It leaves out what a batch
/// slice keeps alive beyond its own rows (a slice of a larger batch pins the
/// whole batch), the encoders' working memory beyond the pages they hold, the
/// distinct values counted to choose a row group's encoding, the bloom
/// filters and page indexes a file keeps until its footer is written, the
/// row groups being assembled or encoded on a writer's own thread, and the
/// pages of a row group beyond its charge that the budget has no room for.
/// Files written with content-defined chunking, and encrypted files, are
/// encoded on the thread that writes them, outside the pool.
///
/// Sharing a pool changes no file contents: a write's row groups close at
/// the same rows and each is encoded from its own rows the same way, on
/// whichever thread, so its files are those of the same write on a pool of
/// its own.
pub struct EncodePool {
    threads: usize,
    budget: Arc<Budget>,
    workers: Mutex<Workers>,
    /// Called with a job's sequence number before it encodes, by every queue
    /// opened after it is set.
    #[cfg(test)]
    before_encode: Mutex<Option<BeforeEncode>>,
}

#[cfg(test)]
type BeforeEncode = Arc<dyn Fn(u64) + Send + Sync>;

struct Workers {
    started: usize,
    jobs: Sender<Job>,
    /// Kept here so the channel stays open while no thread has started.
    queue: Arc<Mutex<Receiver<Job>>>,
}

impl std::fmt::Debug for EncodePool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncodePool")
            .field("threads", &self.threads)
            .field("budget", &self.budget.limit)
            .field("shared", &self.budget.checked)
            .field("reserved", &self.budget.reserved.load(Ordering::Relaxed))
            .field("out", &self.budget.out.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl EncodePool {
    /// A pool of `threads` threads, or `None` for one thread or fewer: the
    /// caller then encodes on its own.
    ///
    /// The budget follows the writer's own row-group size. With a byte cap of
    /// `row_group_bytes` per group it holds two groups per thread, one being
    /// encoded and one waiting to be appended, each counted at its input and
    /// its encoded pages. Without a byte cap it is unlimited, and the count
    /// of row groups out alone bounds what is held: `2 × threads` groups, plus
    /// one for each other queue holding rows (see the module docs), each as
    /// large as the row cap makes it.
    pub(crate) fn new(threads: usize, row_group_bytes: Option<usize>) -> Option<Arc<Self>> {
        let budget = row_group_bytes.map_or(usize::MAX, |bytes| {
            threads.saturating_mul(bytes).saturating_mul(4)
        });
        (threads > 1).then(|| Self::with_budget(threads, budget))
    }

    /// A pool of `threads` threads and a budget of `budget_bytes` bytes, to
    /// share across writes; see the type's docs for how it admits row groups
    /// and what it bounds. One thread is allowed: row groups are then
    /// encoded on it while the writer assembles the next one.
    ///
    /// Threads start as the first row groups arrive, and stop once the last
    /// handle to the pool, and every write holding one, is dropped.
    ///
    /// # Errors
    ///
    /// [`DuckLakeError::InvalidConfig`] when `threads` or `budget_bytes` is 0.
    pub fn try_new(threads: usize, budget_bytes: usize) -> Result<Arc<Self>> {
        if threads == 0 {
            return Err(DuckLakeError::InvalidConfig(
                "an encode pool needs at least one thread".to_string(),
            ));
        }
        if budget_bytes == 0 {
            return Err(DuckLakeError::InvalidConfig(
                "an encode pool needs a budget of at least one byte".to_string(),
            ));
        }
        Ok(Self::build(threads, budget_bytes, true))
    }

    /// A write's own pool of `threads` threads, a budget of `budget` bytes,
    /// and at most `2 × threads` row groups out, admitting loosely (see the
    /// module docs).
    pub(crate) fn with_budget(threads: usize, budget: usize) -> Arc<Self> {
        Self::build(threads, budget, false)
    }

    fn build(threads: usize, budget: usize, checked: bool) -> Arc<Self> {
        let threads = threads.max(1);
        let (jobs, queue) = mpsc::channel();
        Arc::new(Self {
            threads,
            budget: Arc::new(Budget {
                limit: budget,
                checked,
                reserved: AtomicUsize::new(0),
                max_out: threads.saturating_mul(2),
                out: AtomicUsize::new(0),
                #[cfg(test)]
                peak_out: AtomicUsize::new(0),
                #[cfg(test)]
                peak_reserved: AtomicUsize::new(0),
                #[cfg(test)]
                waits: AtomicUsize::new(0),
                #[cfg(test)]
                encoded_here: AtomicUsize::new(0),
            }),
            workers: Mutex::new(Workers {
                started: 0,
                jobs,
                queue: Arc::new(Mutex::new(queue)),
            }),
            #[cfg(test)]
            before_encode: Mutex::new(None),
        })
    }

    /// Run `job` on a pool thread, starting one while fewer than `threads`
    /// have started. With no thread to run it, it runs here.
    fn submit(&self, job: Job) {
        let mut workers = lock(&self.workers);
        if workers.started < self.threads {
            let queue = Arc::clone(&workers.queue);
            // A thread the OS refuses is one thread fewer.
            if std::thread::Builder::new()
                .name("ducklake-encode".to_string())
                .spawn(move || work(&queue))
                .is_ok()
            {
                workers.started += 1;
            }
        }
        if workers.started == 0 {
            drop(workers);
            job();
            return;
        }
        if let Err(mpsc::SendError(job)) = workers.jobs.send(job) {
            drop(workers);
            job();
        }
    }

    fn reserve(&self, bytes: usize) -> Reservation {
        let mut reservation = Reservation {
            budget: Arc::clone(&self.budget),
            bytes: 0,
        };
        reservation.resize(bytes);
        reservation
    }

    /// Call `hook` with each job's sequence number before it encodes, in
    /// every queue opened from now on.
    #[cfg(test)]
    pub(crate) fn set_before_encode(&self, hook: BeforeEncode) {
        *lock(&self.before_encode) = Some(hook);
    }

    /// Count one more row group out, until the slot is dropped.
    fn take_slot(&self) -> OutSlot {
        let _out = self.budget.out.fetch_add(1, Ordering::Relaxed) + 1;
        #[cfg(test)]
        self.budget.peak_out.fetch_max(_out, Ordering::Relaxed);
        OutSlot {
            budget: Arc::clone(&self.budget),
        }
    }

    /// Bytes charged to the budget now.
    #[cfg(test)]
    pub(crate) fn reserved(&self) -> usize {
        self.budget.reserved.load(Ordering::Relaxed)
    }

    /// The most bytes ever charged to the budget at once.
    #[cfg(test)]
    pub(crate) fn peak_reserved(&self) -> usize {
        self.budget.peak_reserved.load(Ordering::Relaxed)
    }

    /// Row groups out now.
    #[cfg(test)]
    pub(crate) fn out(&self) -> usize {
        self.budget.out.load(Ordering::Relaxed)
    }

    /// The most row groups ever out at once.
    #[cfg(test)]
    pub(crate) fn peak_out(&self) -> usize {
        self.budget.peak_out.load(Ordering::Relaxed)
    }

    /// Row groups a shared pool could not take, encoded by their writers.
    #[cfg(test)]
    pub(crate) fn encoded_here(&self) -> usize {
        self.budget.encoded_here.load(Ordering::Relaxed)
    }
}

/// A pool thread: run jobs until the pool is dropped.
fn work(queue: &Mutex<Receiver<Job>>) {
    loop {
        let job = lock(queue).recv();
        match job {
            Ok(job) => job(),
            Err(_) => return,
        }
    }
}

#[derive(Debug)]
struct Budget {
    limit: usize,
    /// Whether row groups are admitted strictly, as a shared pool admits them
    /// ([`RowGroupQueue::dispatch_checked`]), rather than loosely.
    checked: bool,
    reserved: AtomicUsize,
    /// Row groups out across every queue of the pool, past which a queue
    /// with one out appends before it hands out another.
    max_out: usize,
    out: AtomicUsize,
    /// The most row groups ever out at once.
    #[cfg(test)]
    peak_out: AtomicUsize,
    /// The most bytes ever charged at once.
    #[cfg(test)]
    peak_reserved: AtomicUsize,
    /// How many times a queue began waiting for one of its row groups.
    #[cfg(test)]
    waits: AtomicUsize,
    /// Row groups a checked pool could not take, encoded by their queues.
    #[cfg(test)]
    encoded_here: AtomicUsize,
}

impl Budget {
    /// Whether a queue with row groups out appends one before it hands out
    /// another, in a pool that admits loosely.
    fn full(&self) -> bool {
        self.reserved.load(Ordering::Relaxed) > self.limit
            || self.out.load(Ordering::Relaxed) >= self.max_out
    }

    /// Charge `bytes` and count one more row group out, if the charge fits
    /// in what the budget has left and fewer than `max_out` are out; else
    /// take nothing. Each is checked and taken in one atomic step, so
    /// concurrent queues never take past either limit.
    fn try_admit(self: &Arc<Self>, bytes: usize) -> Option<(Reservation, OutSlot)> {
        let _out = self
            .out
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |out| {
                (out < self.max_out).then_some(out + 1)
            })
            .ok()?
            + 1;
        // Dropped, and the count given back, if the charge does not fit.
        let slot = OutSlot {
            budget: Arc::clone(self),
        };
        if !self.try_charge(bytes) {
            return None;
        }
        #[cfg(test)]
        self.peak_out.fetch_max(_out, Ordering::Relaxed);
        Some((
            Reservation {
                budget: Arc::clone(self),
                bytes,
            },
            slot,
        ))
    }

    /// Add `bytes` to the charge if the total stays within the budget.
    fn try_charge(&self, bytes: usize) -> bool {
        let charged =
            self.reserved
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |reserved| {
                    reserved
                        .checked_add(bytes)
                        .filter(|total| *total <= self.limit)
                });
        #[cfg(test)]
        if let Ok(before) = charged {
            self.peak_reserved
                .fetch_max(before + bytes, Ordering::Relaxed);
        }
        charged.is_ok()
    }
}

/// A row group counted out of a pool, released on drop: once its row group
/// is appended, or its job fails.
#[derive(Debug)]
struct OutSlot {
    budget: Arc<Budget>,
}

impl Drop for OutSlot {
    fn drop(&mut self) {
        self.budget.out.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Bytes charged to a pool's budget, released on drop.
#[derive(Debug)]
struct Reservation {
    budget: Arc<Budget>,
    bytes: usize,
}

impl Reservation {
    fn resize(&mut self, bytes: usize) {
        if bytes > self.bytes {
            let _before = self
                .budget
                .reserved
                .fetch_add(bytes - self.bytes, Ordering::Relaxed);
            #[cfg(test)]
            self.budget
                .peak_reserved
                .fetch_max(_before + bytes - self.bytes, Ordering::Relaxed);
        } else {
            self.budget
                .reserved
                .fetch_sub(self.bytes - bytes, Ordering::Relaxed);
        }
        self.bytes = bytes;
    }

    /// Charge what an encoded row group's chunks hold, `bytes`. A checked
    /// pool never grows a charge past its budget: a row group whose chunks
    /// outgrow its charge keeps the charge it had when the budget has no
    /// room for the rest.
    fn settle(&mut self, bytes: usize) {
        if !self.budget.checked || bytes <= self.bytes {
            self.resize(bytes);
        } else if self.budget.try_charge(bytes - self.bytes) {
            self.bytes = bytes;
        }
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.resize(0);
    }
}

/// What every row group of a queue is encoded with.
struct GroupEncoding {
    schema: SchemaRef,
    /// The file's properties, without the pool.
    base: DataFileProperties,
    /// Top-level field `i` owns the leaf column writers `bounds[i]..bounds[i + 1]`.
    bounds: Vec<usize>,
    /// The leading columns whose leaves the [`WrittenLeafStats`] cover.
    stats_columns: usize,
    /// Called with a job's sequence number before it encodes.
    #[cfg(test)]
    before_encode: Option<BeforeEncode>,
}

/// A row group whose column chunks are encoded, ready to append.
pub(crate) struct EncodedRowGroup {
    chunks: Vec<ArrowColumnChunk>,
    rows: usize,
    leaf_stats: WrittenLeafStats,
    /// The chunks' charge, released once they are appended.
    _reservation: Reservation,
    /// The row group's place among the pool's row groups out; `None` for a
    /// row group its queue encoded itself.
    _slot: Option<OutSlot>,
}

/// Encode `batches` as one row group: choose its encoding from its rows,
/// write every leaf in schema order, and note its leaf stats. Returns the
/// closed column chunks, the stats, and the bytes the chunks hold.
fn encode_row_group(
    encoding: &GroupEncoding,
    batches: &[RecordBatch],
) -> Result<(Vec<ArrowColumnChunk>, WrittenLeafStats, usize)> {
    let props = props_for_row_group(&encoding.base, batches)?;
    // A writer over nothing gives the column writers for these properties;
    // the chunks they close are appended to the real file.
    let (_, factory) = ArrowWriter::try_new(std::io::sink(), encoding.schema.clone(), Some(props))?
        .into_serialized_writer()?;
    let mut writers = factory.create_column_writers(0)?;
    let mut leaf_stats = WrittenLeafStats::default();
    let fields = encoding.schema.fields();
    for batch in batches {
        leaf_stats.accumulate(batch, encoding.stats_columns);
        for (i, field) in fields.iter().enumerate() {
            let own = &mut writers[encoding.bounds[i]..encoding.bounds[i + 1]];
            let leaves = compute_leaves(field.as_ref(), batch.column(i))?;
            if leaves.len() != own.len() {
                return Err(DuckLakeError::Internal(format!(
                    "column {} has {} leaves for {} writers",
                    field.name(),
                    leaves.len(),
                    own.len()
                )));
            }
            for (writer, leaf) in own.iter_mut().zip(&leaves) {
                writer.write(leaf)?;
            }
        }
    }
    let bytes = writers.iter().map(ArrowColumnWriter::memory_size).sum();
    let chunks = writers
        .into_iter()
        .map(ArrowColumnWriter::close)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok((chunks, leaf_stats, bytes))
}

/// Encode row group `sequence` of a queue, on a pool thread or the queue's
/// own: [`encode_row_group`], with a panic turned into an error.
fn encode_job(
    encoding: &GroupEncoding,
    sequence: u64,
    batches: &[RecordBatch],
) -> Result<(Vec<ArrowColumnChunk>, WrittenLeafStats, usize)> {
    #[cfg(not(test))]
    let _ = sequence;
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        #[cfg(test)]
        if let Some(hook) = &encoding.before_encode {
            hook(sequence);
        }
        encode_row_group(encoding, batches)
    }))
    .unwrap_or_else(|_| {
        Err(DuckLakeError::Internal(
            "encoding a row group panicked".to_string(),
        ))
    })
}

/// The rows of the row group being assembled.
#[derive(Default)]
struct Assembly {
    batches: Vec<RecordBatch>,
    rows: usize,
    bytes: usize,
    reservation: Option<Reservation>,
}

type Outcome = (u64, Result<EncodedRowGroup>);

/// Where a [`RowGroupQueue`] hands each finished row group, in order.
pub(crate) type Deliver<'a> = dyn FnMut(EncodedRowGroup) -> Result<()> + 'a;

/// Cuts a sequence of batches into row groups, encodes them on an
/// [`EncodePool`], and hands them back in order. See the module docs.
pub(crate) struct RowGroupQueue {
    pool: Arc<EncodePool>,
    encoding: Arc<GroupEncoding>,
    max_rows: Option<usize>,
    max_bytes: Option<usize>,
    assembly: Assembly,
    /// Sequence number of the next row group handed out.
    dispatched: u64,
    /// Sequence number of the next row group to append.
    appended: u64,
    sender: Sender<Outcome>,
    results: Receiver<Outcome>,
    /// Finished row groups that wait for an earlier one.
    done: BTreeMap<u64, Result<EncodedRowGroup>>,
    /// Set once a push or finish failed: the queue's rows are no longer all
    /// in its row groups, so it takes and finishes nothing more.
    failed: bool,
}

impl std::fmt::Debug for RowGroupQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RowGroupQueue")
            .field("pool", &self.pool)
            .field("assembled_rows", &self.assembly.rows)
            .field("out", &(self.dispatched - self.appended))
            .field("failed", &self.failed)
            .finish_non_exhaustive()
    }
}

impl RowGroupQueue {
    /// A queue for files of `schema` written with `props`, whose leaf stats
    /// cover its first `stats_columns` columns; `None` when `props` has no
    /// pool, or a pool cannot write its files ([`supports`]).
    pub(crate) fn try_new(
        schema: SchemaRef,
        props: &DataFileProperties,
        stats_columns: usize,
    ) -> Result<Option<Self>> {
        let Some(pool) = props.pool.clone() else {
            return Ok(None);
        };
        if !supports(&props.props) {
            return Ok(None);
        }
        let descr = parquet::arrow::ArrowSchemaConverter::new()
            .with_coerce_types(props.props.coerce_types())
            .convert(&schema)?;
        let fields = schema.fields().len();
        let mut bounds = vec![0usize; fields + 1];
        for leaf in 0..descr.num_columns() {
            let root = descr.get_column_root_idx(leaf);
            if root >= fields {
                return Err(DuckLakeError::Internal(format!(
                    "leaf column {leaf} belongs to root {root}, past the schema's {fields} fields"
                )));
            }
            bounds[root + 1] += 1;
        }
        for i in 0..fields {
            bounds[i + 1] += bounds[i];
        }
        let (sender, results) = mpsc::channel();
        #[cfg(test)]
        let before_encode = lock(&pool.before_encode).clone();
        Ok(Some(Self {
            encoding: Arc::new(GroupEncoding {
                schema,
                base: DataFileProperties {
                    pool: None,
                    ..props.clone()
                },
                bounds,
                stats_columns,
                #[cfg(test)]
                before_encode,
            }),
            pool,
            max_rows: props.props.max_row_group_row_count(),
            max_bytes: props.props.max_row_group_bytes(),
            assembly: Assembly::default(),
            dispatched: 0,
            appended: 0,
            sender,
            results,
            done: BTreeMap::new(),
            failed: false,
        }))
    }

    /// Whether every row pushed has been handed to a `deliver`.
    pub(crate) fn is_empty(&self) -> bool {
        self.assembly.rows == 0 && self.dispatched == self.appended
    }

    /// Add `batch`'s rows, handing out every row group they complete, and
    /// pass `deliver` each finished row group, in order, that is ready.
    /// Blocks while the pool's budget is spent or it has its most row groups
    /// out, appending this queue's earliest one; a shared pool's queue with
    /// none out encodes a row group the pool cannot take on this thread.
    ///
    /// An error fails the queue: every later push and finish returns one.
    pub(crate) fn push(&mut self, batch: &RecordBatch, deliver: &mut Deliver<'_>) -> Result<()> {
        self.check()?;
        let pushed = self.push_rows(batch, deliver);
        self.failed |= pushed.is_err();
        pushed
    }

    fn push_rows(&mut self, batch: &RecordBatch, deliver: &mut Deliver<'_>) -> Result<()> {
        let mut rest = batch.clone();
        while rest.num_rows() > 0 {
            let take = self.rows_to_take(&rest);
            if take == 0 {
                self.dispatch(deliver)?;
                continue;
            }
            let head = rest.slice(0, take);
            rest = rest.slice(take, rest.num_rows() - take);
            let bytes = sliced_memory_size(&head);
            let assembly = &mut self.assembly;
            assembly.rows += head.num_rows();
            assembly.bytes += bytes;
            assembly.batches.push(head);
            let total = assembly.bytes;
            // A checked pool charges a row group when it is handed out.
            if !self.pool.budget.checked {
                assembly
                    .reservation
                    .get_or_insert_with(|| self.pool.reserve(0))
                    .resize(total);
            }
            let full = self.max_rows.is_some_and(|max| assembly.rows >= max)
                || self.max_bytes.is_some_and(|max| assembly.bytes >= max);
            if full {
                self.dispatch(deliver)?;
            }
        }
        self.deliver_ready(deliver)
    }

    /// Hand out the row group being assembled, then wait for every row group
    /// out and pass each to `deliver`, in order. Fails, appending nothing,
    /// once the queue has failed.
    pub(crate) fn finish(&mut self, deliver: &mut Deliver<'_>) -> Result<()> {
        self.check()?;
        let finished = self.finish_rows(deliver);
        self.failed |= finished.is_err();
        finished
    }

    fn finish_rows(&mut self, deliver: &mut Deliver<'_>) -> Result<()> {
        self.dispatch(deliver)?;
        while self.appended < self.dispatched {
            self.deliver_next(deliver)?;
        }
        Ok(())
    }

    /// Refuse to go on once an earlier push or finish failed.
    fn check(&self) -> Result<()> {
        if self.failed {
            return Err(DuckLakeError::Internal(
                "an earlier row group of this data file failed to encode or append".to_string(),
            ));
        }
        Ok(())
    }

    /// How many of `batch`'s leading rows go to the row group being
    /// assembled; 0 when that row group is to be handed out first.
    fn rows_to_take(&self, batch: &RecordBatch) -> usize {
        let rows = batch.num_rows();
        let assembled = self.assembly.rows;
        let mut take = rows;
        if let Some(max) = self.max_rows {
            take = take.min(max.saturating_sub(assembled));
        }
        if let Some(max) = self.max_bytes {
            let room = max.saturating_sub(self.assembly.bytes);
            let bytes = sliced_memory_size(batch);
            // The rows that fit, at the batch's average row size.
            let fit = if bytes == 0 {
                rows
            } else {
                usize::try_from(room as u128 * rows as u128 / bytes as u128).unwrap_or(rows)
            };
            take = take.min(fit);
        }
        if take == 0 && assembled == 0 {
            // A row larger than the byte cap is a row group of its own.
            take = 1;
        }
        take
    }

    /// Hand the row group being assembled to the pool, first appending this
    /// queue's earliest row groups out while the pool's budget is spent or it
    /// has its most row groups out. A queue with none out hands its row group
    /// out at once: it would otherwise wait on another queue's. A pool that
    /// admits strictly takes it only once it fits
    /// ([`Self::dispatch_checked`]).
    fn dispatch(&mut self, deliver: &mut Deliver<'_>) -> Result<()> {
        let Assembly {
            batches,
            rows,
            bytes,
            reservation,
        } = std::mem::take(&mut self.assembly);
        if rows == 0 {
            return Ok(());
        }
        if self.pool.budget.checked {
            return self.dispatch_checked(batches, rows, bytes, deliver);
        }
        let mut reservation = reservation.unwrap_or_else(|| self.pool.reserve(0));
        // The rows, and as much again for the pages they encode to.
        reservation.resize(bytes.saturating_mul(2));
        while self.appended < self.dispatched && self.pool.budget.full() {
            self.deliver_next(deliver)?;
        }
        let slot = self.pool.take_slot();
        self.submit(batches, rows, reservation, slot);
        Ok(())
    }

    /// Hand a row group of `rows` rows, `bytes` in memory, to a pool that
    /// admits strictly, once its charge fits: twice `bytes`, for the rows and
    /// the pages they encode to, within what the budget has left, and one
    /// more row group out within the pool's most. Until it fits, append this
    /// queue's earliest row group out, which gives its charge back; once the
    /// queue has none out, encode the row group here, on the calling thread,
    /// and append it at once. So the queue never waits on another queue's
    /// row groups, a write is never refused for want of budget, and the row
    /// group encoded here is the queue's only one held outside the pool.
    fn dispatch_checked(
        &mut self,
        batches: Vec<RecordBatch>,
        rows: usize,
        bytes: usize,
        deliver: &mut Deliver<'_>,
    ) -> Result<()> {
        let charge = bytes.saturating_mul(2);
        loop {
            if let Some((reservation, slot)) = self.pool.budget.try_admit(charge) {
                self.submit(batches, rows, reservation, slot);
                return Ok(());
            }
            if self.appended == self.dispatched {
                break;
            }
            self.deliver_next(deliver)?;
        }
        #[cfg(test)]
        self.pool
            .budget
            .encoded_here
            .fetch_add(1, Ordering::Relaxed);
        let sequence = self.dispatched;
        self.dispatched += 1;
        let encoded = encode_job(&self.encoding, sequence, &batches);
        drop(batches);
        // Nothing of the queue's is out, so this row group is next in order.
        self.appended += 1;
        let (chunks, leaf_stats, _) = encoded?;
        deliver(EncodedRowGroup {
            chunks,
            rows,
            leaf_stats,
            _reservation: self.pool.reserve(0),
            _slot: None,
        })
    }

    /// Hand `batches`, charged `reservation` and counted out by `slot`, to
    /// the pool as the queue's next row group.
    fn submit(
        &mut self,
        batches: Vec<RecordBatch>,
        rows: usize,
        mut reservation: Reservation,
        slot: OutSlot,
    ) {
        let sequence = self.dispatched;
        self.dispatched += 1;
        let encoding = Arc::clone(&self.encoding);
        let sender = self.sender.clone();
        self.pool.submit(Box::new(move || {
            let encoded = encode_job(&encoding, sequence, &batches);
            drop(batches);
            let outcome = encoded.map(|(chunks, leaf_stats, bytes)| {
                reservation.settle(bytes);
                EncodedRowGroup {
                    chunks,
                    rows,
                    leaf_stats,
                    _reservation: reservation,
                    _slot: Some(slot),
                }
            });
            // A queue dropped mid-write no longer wants it.
            let _ = sender.send((sequence, outcome));
        }));
    }

    /// Pass `deliver` the finished row groups that are next in order, without
    /// waiting for any.
    fn deliver_ready(&mut self, deliver: &mut Deliver<'_>) -> Result<()> {
        while let Ok((sequence, outcome)) = self.results.try_recv() {
            self.done.insert(sequence, outcome);
        }
        while let Some(outcome) = self.done.remove(&self.appended) {
            self.appended += 1;
            deliver(outcome?)?;
        }
        Ok(())
    }

    /// Wait for the earliest row group out and pass it to `deliver`.
    fn deliver_next(&mut self, deliver: &mut Deliver<'_>) -> Result<()> {
        loop {
            if let Some(outcome) = self.done.remove(&self.appended) {
                self.appended += 1;
                return deliver(outcome?);
            }
            #[cfg(test)]
            if let Ok((sequence, outcome)) = self.results.try_recv() {
                self.done.insert(sequence, outcome);
                continue;
            }
            #[cfg(test)]
            self.pool.budget.waits.fetch_add(1, Ordering::Relaxed);
            // The queue holds a sender, so this waits for a job rather than
            // failing; every job sends, a panicking one included.
            let (sequence, outcome) = self.results.recv().map_err(|_| {
                DuckLakeError::Internal("the row-group encoder stopped".to_string())
            })?;
            self.done.insert(sequence, outcome);
        }
    }
}

/// A Parquet file built from encoded row groups, appended in order.
pub(crate) struct RowGroupAppender<W: Write + Send> {
    writer: SerializedFileWriter<W>,
}

impl<W: Write + Send> std::fmt::Debug for RowGroupAppender<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RowGroupAppender")
            .field("bytes_written", &self.writer.bytes_written())
            .field("row_groups", &self.writer.flushed_row_groups().len())
            .finish()
    }
}

impl<W: Write + Send> RowGroupAppender<W> {
    /// Open a file of `schema` on `sink`. `ArrowWriter` opens it, so the
    /// magic, the schema and the Arrow schema in the footer are the ones it
    /// writes; `props` decides what is file-wide.
    pub(crate) fn try_new(sink: W, schema: SchemaRef, props: &WriterProperties) -> Result<Self> {
        let (writer, _) =
            ArrowWriter::try_new(sink, schema, Some(props.clone()))?.into_serialized_writer()?;
        Ok(Self {
            writer,
        })
    }

    /// Append `group` as the file's next row group, returning its rows and
    /// leaf stats.
    pub(crate) fn append(&mut self, group: EncodedRowGroup) -> Result<(usize, WrittenLeafStats)> {
        let EncodedRowGroup {
            chunks,
            rows,
            leaf_stats,
            _reservation,
            _slot,
        } = group;
        let mut row_group = self.writer.next_row_group()?;
        for chunk in chunks {
            chunk.append_to_row_group(&mut row_group)?;
        }
        row_group.close()?;
        Ok((rows, leaf_stats))
    }

    /// Bytes written to the sink so far.
    pub(crate) fn bytes_written(&self) -> usize {
        self.writer.bytes_written()
    }

    /// Write the footer and return the sink.
    pub(crate) fn into_inner(self) -> Result<W> {
        Ok(self.writer.into_inner()?)
    }
}

/// One data file whose row groups are encoded on a pool, with the leaf stats
/// of every row appended.
pub(crate) struct ParallelFileWriter<W: Write + Send> {
    queue: RowGroupQueue,
    file: RowGroupAppender<W>,
    leaf_stats: WrittenLeafStats,
}

impl<W: Write + Send> std::fmt::Debug for ParallelFileWriter<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelFileWriter")
            .field("queue", &self.queue)
            .field("file", &self.file)
            .finish_non_exhaustive()
    }
}

impl<W: Write + Send> ParallelFileWriter<W> {
    /// Open a file of `schema` on `sink`, encoding its row groups on `queue`.
    pub(crate) fn try_new(
        sink: W,
        schema: SchemaRef,
        props: &WriterProperties,
        queue: RowGroupQueue,
    ) -> Result<Self> {
        Ok(Self {
            queue,
            file: RowGroupAppender::try_new(sink, schema, props)?,
            leaf_stats: WrittenLeafStats::default(),
        })
    }

    pub(crate) fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        let Self {
            queue,
            file,
            leaf_stats,
        } = self;
        queue.push(batch, &mut |group| append_to(file, leaf_stats, group))
    }

    /// Append every row group, write the footer, and return the sink and the
    /// file's leaf stats.
    pub(crate) fn into_inner(mut self) -> Result<(W, WrittenLeafStats)> {
        let Self {
            queue,
            file,
            leaf_stats,
        } = &mut self;
        queue.finish(&mut |group| append_to(file, leaf_stats, group))?;
        Ok((self.file.into_inner()?, self.leaf_stats))
    }
}

fn append_to<W: Write + Send>(
    file: &mut RowGroupAppender<W>,
    leaf_stats: &mut WrittenLeafStats,
    group: EncodedRowGroup,
) -> Result<()> {
    let (_, stats) = file.append(group)?;
    leaf_stats.merge_ordered(&stats);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{
        ArrayRef, BooleanArray, Float64Array, Int64Array, ListArray, MapArray, StringArray,
        StructArray, TimestampNanosecondArray,
    };
    use arrow::buffer::{NullBuffer, OffsetBuffer};
    use arrow::datatypes::{DataType, Field, Fields, Schema, TimeUnit};
    use bytes::Bytes;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::basic::Compression;
    use parquet::file::reader::{FileReader, SerializedFileReader};

    use super::*;

    /// A schema of every shape a leaf comes from: a timestamp, scalars, a
    /// long text column, maps of strings and of floats, a list and a struct.
    fn schema() -> SchemaRef {
        let entries = |value: DataType| {
            Arc::new(Field::new(
                "entries",
                DataType::Struct(Fields::from(vec![
                    Field::new("key", DataType::Utf8, false),
                    Field::new("value", value, true),
                ])),
                false,
            ))
        };
        Arc::new(Schema::new(vec![
            Field::new("ts", DataType::Timestamp(TimeUnit::Nanosecond, None), true),
            Field::new("id", DataType::Int64, false),
            Field::new("status", DataType::Utf8, true),
            Field::new("body", DataType::Utf8, true),
            Field::new("attrs", DataType::Map(entries(DataType::Utf8), false), true),
            Field::new(
                "nums",
                DataType::Map(entries(DataType::Float64), false),
                true,
            ),
            Field::new(
                "tags",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
            Field::new(
                "point",
                DataType::Struct(Fields::from(vec![
                    Field::new("x", DataType::Float64, true),
                    Field::new("ok", DataType::Boolean, true),
                ])),
                true,
            ),
        ]))
    }

    fn map(keys: Vec<String>, values: ArrayRef, lengths: &[usize], nulls: NullBuffer) -> ArrayRef {
        let value_type = values.data_type().clone();
        let fields = Fields::from(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", value_type, true),
        ]);
        Arc::new(MapArray::new(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(fields.clone()),
                false,
            )),
            OffsetBuffer::from_lengths(lengths.iter().copied()),
            StructArray::new(
                fields,
                vec![Arc::new(StringArray::from(keys)), values],
                None,
            ),
            Some(nulls),
            false,
        ))
    }

    /// Rows `lo..hi`: repeating and unique values, NULLs in every nullable
    /// column, maps of zero to three entries, bodies of up to 2 KB.
    fn batch(lo: usize, hi: usize) -> RecordBatch {
        let rows = lo..hi;
        let lengths: Vec<usize> = rows.clone().map(|i| i % 4).collect();
        let total: usize = lengths.iter().sum();
        let valid = NullBuffer::from_iter(rows.clone().map(|i| i % 11 != 0));
        let attrs = map(
            (0..total).map(|k| format!("attr.key.{}", k % 7)).collect(),
            Arc::new(StringArray::from_iter(
                (0..total).map(|k| (k % 5 != 0).then(|| format!("value-{}", k % 13))),
            )),
            &lengths,
            valid.clone(),
        );
        let nums = map(
            (0..total).map(|k| format!("metric.{}", k % 3)).collect(),
            Arc::new(Float64Array::from_iter(
                (0..total).map(|k| (k % 6 != 0).then_some(k as f64 * 0.25)),
            )),
            &lengths,
            valid,
        );
        // One tag per row, or a NULL list.
        let tags = ListArray::new(
            Arc::new(Field::new("item", DataType::Utf8, true)),
            OffsetBuffer::from_lengths(rows.clone().map(|i| usize::from(i % 9 != 0))),
            Arc::new(StringArray::from_iter_values(
                rows.clone()
                    .filter(|i| i % 9 != 0)
                    .map(|i| format!("tag-{}", i % 4)),
            )),
            Some(NullBuffer::from_iter(rows.clone().map(|i| i % 9 != 0))),
        );
        let point = StructArray::new(
            Fields::from(vec![
                Field::new("x", DataType::Float64, true),
                Field::new("ok", DataType::Boolean, true),
            ]),
            vec![
                Arc::new(Float64Array::from_iter(
                    rows.clone().map(|i| (i % 3 != 0).then_some(i as f64)),
                )),
                Arc::new(BooleanArray::from_iter(
                    rows.clone().map(|i| (i % 5 != 0).then_some(i % 2 == 0)),
                )),
            ],
            Some(NullBuffer::from_iter(rows.clone().map(|i| i % 17 != 0))),
        );
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(TimestampNanosecondArray::from_iter(
                    rows.clone()
                        .map(|i| (i % 19 != 0).then_some(i as i64 * 1_000_000)),
                )),
                Arc::new(Int64Array::from_iter_values(rows.clone().map(|i| i as i64))),
                Arc::new(StringArray::from_iter(rows.clone().map(|i| {
                    (i % 7 != 0).then(|| ["Ok", "Unset", "Error"][i % 3].to_string())
                }))),
                Arc::new(StringArray::from_iter(rows.clone().map(|i| {
                    (i % 13 != 0).then(|| format!("{i}:{}", "x".repeat(i * 37 % 2_000)))
                }))),
                attrs,
                nums,
                Arc::new(tags),
                Arc::new(point),
            ],
        )
        .unwrap()
    }

    /// 25,337 rows in batches of uneven sizes, one larger than every row
    /// cap used.
    fn batches() -> Vec<RecordBatch> {
        let mut out = Vec::new();
        let mut lo = 0;
        for size in [1, 4_000, 333, 7_001, 2, 5_000, 9_000] {
            out.push(batch(lo, lo + size));
            lo += size;
        }
        out
    }

    fn writer_props(rows: Option<usize>, bytes: Option<usize>) -> WriterProperties {
        WriterProperties::builder()
            .set_compression(Compression::LZ4)
            .set_max_row_group_row_count(rows)
            .set_max_row_group_bytes(bytes)
            .build()
    }

    fn props(
        rows: Option<usize>,
        bytes: Option<usize>,
        pool: Arc<EncodePool>,
    ) -> DataFileProperties {
        DataFileProperties {
            props: writer_props(rows, bytes),
            dictionary_set: Arc::default(),
            bloom_filter_set: Arc::default(),
            pool: Some(pool),
        }
    }

    fn queue(props: &DataFileProperties) -> RowGroupQueue {
        RowGroupQueue::try_new(schema(), props, schema().fields().len())
            .unwrap()
            .expect("a pool that supports the properties")
    }

    /// Write `batches` through a [`ParallelFileWriter`].
    fn parallel(
        batches: &[RecordBatch],
        props: &DataFileProperties,
    ) -> (Vec<u8>, WrittenLeafStats) {
        let mut writer =
            ParallelFileWriter::try_new(Vec::new(), schema(), &props.props, queue(props)).unwrap();
        for b in batches {
            writer.write(b).unwrap();
        }
        writer.into_inner().unwrap()
    }

    fn read(bytes: Vec<u8>) -> RecordBatch {
        let reader = ParquetRecordBatchReaderBuilder::try_new(Bytes::from(bytes))
            .unwrap()
            .build()
            .unwrap();
        let read: Vec<RecordBatch> = reader.map(|b| b.unwrap()).collect();
        arrow::compute::concat_batches(&read[0].schema(), &read).unwrap()
    }

    fn row_group_rows(bytes: &[u8]) -> Vec<i64> {
        let reader = SerializedFileReader::new(Bytes::copy_from_slice(bytes)).unwrap();
        reader
            .metadata()
            .row_groups()
            .iter()
            .map(|g| g.num_rows())
            .collect()
    }

    fn assert_same_rows(got: &RecordBatch, batches: &[RecordBatch]) {
        let expected = arrow::compute::concat_batches(&schema(), batches).unwrap();
        assert_eq!(got.num_rows(), expected.num_rows());
        for (g, e) in got.columns().iter().zip(expected.columns()) {
            assert_eq!(g.to_data(), e.to_data());
        }
    }

    /// The values read back are the ones written, in order, at any thread
    /// count and row cap; the row cap cuts row groups exactly as
    /// `ArrowWriter` cuts them; and the leaf stats are those of one pass over
    /// every row.
    #[test]
    fn row_groups_encoded_on_threads_read_back_as_written() {
        let batches = batches();
        let mut sequential = WrittenLeafStats::default();
        for b in &batches {
            sequential.accumulate(b, schema().fields().len());
        }
        for rows in [Some(97), Some(3_000), Some(1 << 20)] {
            let mut serial =
                ArrowWriter::try_new(Vec::new(), schema(), Some(writer_props(rows, None))).unwrap();
            for b in &batches {
                serial.write(b).unwrap();
            }
            let serial = serial.into_inner().unwrap();
            for threads in [2, 3, 8] {
                let pool = EncodePool::new(threads, None).unwrap();
                let (bytes, stats) = parallel(&batches, &props(rows, None, pool));
                assert_eq!(
                    row_group_rows(&bytes),
                    row_group_rows(&serial),
                    "rows {rows:?}"
                );
                assert_same_rows(&read(bytes), &batches);
                assert_eq!(format!("{stats:?}"), format!("{sequential:?}"));
            }
        }
    }

    /// A bloom filter the caller set holds in every row group encoded on a
    /// thread, whatever that row group's dictionary decision.
    #[test]
    fn a_bloom_filter_the_caller_set_reaches_every_row_group_encoded_on_threads() {
        let id = parquet::schema::types::ColumnPath::from("id");
        let base = props(Some(3_000), None, EncodePool::new(3, None).unwrap());
        let props = DataFileProperties {
            props: base
                .props
                .clone()
                .into_builder()
                .set_column_bloom_filter_enabled(id.clone(), true)
                .set_column_bloom_filter_fpp(id, crate::write_encoding::BLOOM_FILTER_FPP)
                .build(),
            bloom_filter_set: Arc::new(std::collections::HashSet::from([
                parquet::schema::types::ColumnPath::from("id"),
            ])),
            ..base
        };
        let (bytes, _) = parallel(&batches(), &props);
        let reader = SerializedFileReader::new(Bytes::from(bytes)).unwrap();
        let groups = reader.metadata().row_groups();
        assert!(groups.len() > 1, "{} row groups", groups.len());
        for (g, group) in groups.iter().enumerate() {
            // `id` is unique, so the rule keeps no dictionary and no filter.
            let id = group.column(1);
            assert_eq!(id.column_path().string(), "id");
            assert!(id.bloom_filter_offset().is_some(), "row group {g}");
            assert!(id.dictionary_page_offset().is_none(), "row group {g}");
        }
    }

    /// A byte cap closes a row group once its input reaches it, splitting a
    /// batch where it does; a row larger than the cap is a row group alone.
    #[test]
    fn a_byte_cap_closes_row_groups_by_their_input() {
        let batches = batches();
        let pool = EncodePool::new(4, None).unwrap();
        let (bytes, _) = parallel(&batches, &props(None, Some(256 * 1024), pool));
        let groups = row_group_rows(&bytes);
        assert!(groups.len() > 10, "{groups:?}");
        assert_same_rows(&read(bytes), &batches);

        // Rows of about 2 KB each against a 1 KB cap.
        let wide = batch(1_000, 1_100);
        let pool = EncodePool::new(4, None).unwrap();
        let (bytes, _) = parallel(std::slice::from_ref(&wide), &props(None, Some(1_024), pool));
        let groups = row_group_rows(&bytes);
        assert!(groups.iter().all(|rows| *rows <= 2), "{groups:?}");
        assert_eq!(groups.iter().sum::<i64>(), 100);
        assert_same_rows(&read(bytes), &[wide]);
    }

    /// A budget smaller than one row group still writes every row: the queue
    /// hands out one row group at a time, and gives every byte back.
    #[test]
    fn a_budget_smaller_than_a_row_group_writes_one_at_a_time() {
        let batches = batches();
        let pool = EncodePool::with_budget(4, 1);
        let (bytes, _) = parallel(&batches, &props(Some(1_000), None, Arc::clone(&pool)));
        assert_eq!(row_group_rows(&bytes).len(), 26);
        assert_same_rows(&read(bytes), &batches);
        assert_eq!(pool.budget.reserved.load(Ordering::Relaxed), 0);
        assert_eq!(pool.budget.out.load(Ordering::Relaxed), 0);
    }

    /// The count of row groups out is the pool's, not each queue's: with
    /// many queues on one pool, as a partitioned write has one per open
    /// partition, a queue with a row group out waits for it once the pool
    /// has `2 × threads` out, and only a queue with none out hands one out
    /// past that. Every row group is still written, and every count is given
    /// back.
    #[test]
    fn row_groups_out_are_bounded_across_every_queue_of_a_pool() {
        const QUEUES: usize = 20;
        let threads = 2;
        let pool = EncodePool::new(threads, None).unwrap();
        // Every job waits until the gate opens.
        let gate = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
        let hook_gate = Arc::clone(&gate);
        *lock(&pool.before_encode) = Some(Arc::new(move |_| {
            let (open, opened) = &*hook_gate;
            let mut open = lock(open);
            while !*open {
                open = opened.wait(open).unwrap_or_else(|p| p.into_inner());
            }
        }));
        let props = props(Some(100), None, Arc::clone(&pool));
        let mut queues: Vec<RowGroupQueue> = (0..QUEUES).map(|_| queue(&props)).collect();

        // Open the gate once a queue waits for a row group of its own, and
        // note how many were out then.
        let opener = {
            let pool = Arc::clone(&pool);
            let gate = Arc::clone(&gate);
            std::thread::spawn(move || {
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
                while pool.budget.waits.load(Ordering::Relaxed) == 0 {
                    assert!(std::time::Instant::now() < deadline, "no queue ever waited");
                    std::thread::yield_now();
                }
                let out = pool.budget.out.load(Ordering::Relaxed);
                *lock(&gate.0) = true;
                gate.1.notify_all();
                out
            })
        };

        let mut appended = vec![0usize; QUEUES];
        let rounds = 4;
        for round in 0..rounds {
            for (i, queue) in queues.iter_mut().enumerate() {
                let lo = (round * QUEUES + i) * 100;
                let count = &mut appended[i];
                queue
                    .push(&batch(lo, lo + 100), &mut |_| {
                        *count += 1;
                        Ok(())
                    })
                    .unwrap();
            }
        }
        for (i, queue) in queues.iter_mut().enumerate() {
            let count = &mut appended[i];
            queue
                .finish(&mut |_| {
                    *count += 1;
                    Ok(())
                })
                .unwrap();
        }
        // The first round hands out one row group a queue: the first
        // `2 × threads` under the cap, every other because its queue had
        // none out. The first queue's second row group then waits.
        assert_eq!(opener.join().unwrap(), QUEUES);
        let peak = pool.budget.peak_out.load(Ordering::Relaxed);
        assert!(
            peak <= 2 * threads + QUEUES,
            "{peak} row groups were out at once"
        );
        assert_eq!(appended, vec![rounds; QUEUES]);
        assert_eq!(pool.budget.out.load(Ordering::Relaxed), 0);
        assert_eq!(pool.budget.reserved.load(Ordering::Relaxed), 0);
    }

    /// A row group whose job finishes after a later one's is still appended
    /// first.
    #[test]
    fn row_groups_are_appended_in_order_whichever_finishes_first() {
        let (release, released) = mpsc::channel::<()>();
        let released = Mutex::new(released);
        let props = props(Some(1_000), None, EncodePool::new(2, None).unwrap());
        let mut queue = queue(&props);
        Arc::get_mut(&mut queue.encoding).unwrap().before_encode =
            Some(Arc::new(move |sequence| {
                if sequence == 0 {
                    lock(&released).recv().unwrap();
                }
            }));
        let mut file = RowGroupAppender::try_new(Vec::new(), schema(), &props.props).unwrap();
        let mut appended = 0;
        let mut deliver = |group: EncodedRowGroup| {
            file.append(group)?;
            appended += 1;
            Ok(())
        };
        let input = batch(0, 2_000);
        queue.push(&input, &mut deliver).unwrap();
        // Row group 1 finishes while row group 0 waits, and is held back.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while !queue.done.contains_key(&1) {
            assert!(
                std::time::Instant::now() < deadline,
                "row group 1 never finished"
            );
            if let Ok((sequence, outcome)) = queue
                .results
                .recv_timeout(std::time::Duration::from_millis(50))
            {
                queue.done.insert(sequence, outcome);
            }
        }
        queue.deliver_ready(&mut deliver).unwrap();
        assert!(queue.done.contains_key(&1), "row group 1 went first");
        release.send(()).unwrap();
        queue.finish(&mut deliver).unwrap();
        assert_eq!(appended, 2);
        assert!(queue.is_empty());
        let bytes = file.into_inner().unwrap();
        assert_eq!(row_group_rows(&bytes), vec![1_000, 1_000]);
        assert_same_rows(&read(bytes), &[input]);
    }

    /// A failed row group fails the write, and nothing after it is appended.
    #[test]
    fn a_failed_row_group_fails_the_write() {
        let props = props(Some(1_000), None, EncodePool::new(2, None).unwrap());
        let mut queue = queue(&props);
        Arc::get_mut(&mut queue.encoding).unwrap().before_encode = Some(Arc::new(|sequence| {
            if sequence == 1 {
                panic!("row group {sequence} failed");
            }
        }));
        let mut appended = 0;
        let mut deliver = |_: EncodedRowGroup| {
            appended += 1;
            Ok(())
        };
        let pushed = queue.push(&batch(0, 3_000), &mut deliver);
        let error = pushed
            .and_then(|()| queue.finish(&mut deliver))
            .expect_err("a failed row group fails the write");
        assert!(error.to_string().contains("panicked"), "{error}");
        assert!(appended <= 1, "{appended} row groups appended");
    }

    /// A row group that fails while rows are still being pushed fails the
    /// queue for good: the push that sees it fails, and so does every later
    /// push and the finish, which appends none of the row groups after it.
    #[test]
    fn a_failed_row_group_fails_every_later_push_and_the_finish() {
        let pool = EncodePool::new(2, None).unwrap();
        let props = props(Some(100), None, Arc::clone(&pool));
        let mut queue = queue(&props);
        Arc::get_mut(&mut queue.encoding).unwrap().before_encode = Some(Arc::new(|sequence| {
            if sequence == 0 {
                panic!("row group {sequence} failed");
            }
        }));
        let mut appended = 0;
        let mut deliver = |_: EncodedRowGroup| {
            appended += 1;
            Ok(())
        };
        // A push fails at the latest once the pool is full and the queue
        // must append its first row group.
        let mut pushes = 0;
        loop {
            pushes += 1;
            assert!(pushes <= 100, "no push saw the failed row group");
            let lo = pushes * 100;
            if queue.push(&batch(lo, lo + 100), &mut deliver).is_err() {
                break;
            }
        }
        let error = queue
            .push(&batch(0, 100), &mut deliver)
            .expect_err("a failed queue takes no more rows");
        assert!(error.to_string().contains("earlier row group"), "{error}");
        let error = queue
            .finish(&mut deliver)
            .expect_err("a failed queue does not finish");
        assert!(error.to_string().contains("earlier row group"), "{error}");
        assert_eq!(appended, 0, "a row group after the failed one was appended");
        drop(queue);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while pool.budget.out.load(Ordering::Relaxed) != 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "the row groups out were never given back"
            );
            std::thread::yield_now();
        }
    }

    #[test]
    fn a_shared_pool_needs_a_thread_and_a_budget() {
        let error = EncodePool::try_new(0, 1 << 20).expect_err("no threads");
        assert!(matches!(error, DuckLakeError::InvalidConfig(_)), "{error}");
        let error = EncodePool::try_new(2, 0).expect_err("no budget");
        assert!(matches!(error, DuckLakeError::InvalidConfig(_)), "{error}");
        let pool = EncodePool::try_new(1, 1).unwrap();
        assert!(pool.budget.checked);
        assert_eq!(pool.budget.max_out, 2);
    }

    /// The file `batches` makes on a write's own pool, at a row cap of
    /// `rows`: what a shared pool must make too.
    fn own_pool_file(batches: &[RecordBatch], rows: usize) -> Vec<u8> {
        parallel(
            batches,
            &props(Some(rows), None, EncodePool::new(2, None).unwrap()),
        )
        .0
    }

    /// Writes sharing a pool whose budget holds about two of their row
    /// groups all write every row, in order, into the files each writes on
    /// a pool of its own: the charges out never pass the budget, nor the
    /// row groups out `2 × threads`, and every byte and count is given back.
    #[test]
    fn writes_sharing_a_pool_stay_within_its_budget() {
        const WRITES: usize = 4;
        const ROWS: usize = 500;
        let batches = batches();
        let group = batches
            .iter()
            .map(sliced_memory_size)
            .sum::<usize>()
            .div_ceil(batches.iter().map(RecordBatch::num_rows).sum::<usize>())
            * ROWS;
        // Two row groups' charges, each twice its rows.
        let budget = 4 * group;
        let threads = 2;
        let pool = EncodePool::try_new(threads, budget).unwrap();
        let expected = own_pool_file(&batches, ROWS);
        let start = std::sync::Barrier::new(WRITES);
        let files: Vec<Vec<u8>> = std::thread::scope(|scope| {
            let writers: Vec<_> = (0..WRITES)
                .map(|_| {
                    let props = props(Some(ROWS), None, Arc::clone(&pool));
                    let (batches, start) = (&batches, &start);
                    scope.spawn(move || {
                        start.wait();
                        parallel(batches, &props).0
                    })
                })
                .collect();
            writers.into_iter().map(|w| w.join().unwrap()).collect()
        });
        for file in files {
            assert!(file == expected, "a shared pool changed a file's bytes");
        }
        let peak = pool.peak_reserved();
        assert!(peak > 0, "no row group went to the pool");
        assert!(peak <= budget, "{peak} bytes charged of {budget}");
        assert!(pool.peak_out() <= 2 * threads, "{}", pool.peak_out());
        assert_eq!(pool.reserved(), 0);
        assert_eq!(pool.out(), 0);
    }

    /// A shared pool whose budget is smaller than one row group takes none:
    /// the writer encodes each on its own thread, and writes the file a
    /// write's own pool writes.
    #[test]
    fn a_shared_budget_smaller_than_a_row_group_encodes_each_on_the_writing_thread() {
        let batches = batches();
        let pool = EncodePool::try_new(4, 1).unwrap();
        let (bytes, _) = parallel(&batches, &props(Some(1_000), None, Arc::clone(&pool)));
        assert_eq!(row_group_rows(&bytes).len(), 26);
        assert_eq!(pool.encoded_here(), 26);
        assert_eq!(pool.peak_out(), 0);
        assert_eq!(pool.peak_reserved(), 0);
        assert!(bytes == own_pool_file(&batches, 1_000));
        assert_same_rows(&read(bytes), &batches);
    }

    /// A shared pool of one thread writes every row, its row groups cut as
    /// `ArrowWriter` cuts them.
    #[test]
    fn a_shared_pool_of_one_thread_writes_every_row() {
        let batches = batches();
        let mut serial =
            ArrowWriter::try_new(Vec::new(), schema(), Some(writer_props(Some(3_000), None)))
                .unwrap();
        for b in &batches {
            serial.write(b).unwrap();
        }
        let serial = serial.into_inner().unwrap();
        let pool = EncodePool::try_new(1, usize::MAX).unwrap();
        let (bytes, _) = parallel(&batches, &props(Some(3_000), None, Arc::clone(&pool)));
        assert_eq!(row_group_rows(&bytes), row_group_rows(&serial));
        assert_same_rows(&read(bytes), &batches);
        assert_eq!(pool.encoded_here(), 0);
        assert!(pool.peak_out() <= 2, "{}", pool.peak_out());
        assert_eq!(pool.reserved(), 0);
    }

    /// Wait until `pool` has nothing charged and nothing out.
    fn wait_until_given_back(pool: &EncodePool) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while pool.reserved() != 0 || pool.out() != 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "{} bytes and {} row groups were never given back",
                pool.reserved(),
                pool.out()
            );
            std::thread::yield_now();
        }
    }

    /// A row group that fails on a shared pool fails the write, and every
    /// byte and count of the write's row groups is given back.
    #[test]
    fn a_failed_row_group_gives_back_a_shared_pools_budget() {
        let pool = EncodePool::try_new(2, usize::MAX).unwrap();
        pool.set_before_encode(Arc::new(|sequence| {
            if sequence == 1 {
                panic!("row group {sequence} failed");
            }
        }));
        let props = props(Some(1_000), None, Arc::clone(&pool));
        let mut queue = queue(&props);
        let mut deliver = |_: EncodedRowGroup| Ok(());
        let error = queue
            .push(&batch(0, 3_000), &mut deliver)
            .and_then(|()| queue.finish(&mut deliver))
            .expect_err("a failed row group fails the write");
        assert!(error.to_string().contains("panicked"), "{error}");
        drop(queue);
        wait_until_given_back(&pool);
    }

    /// A row group that fails while its writer encodes it fails the write,
    /// with nothing charged.
    #[test]
    fn a_row_group_failed_on_the_writing_thread_fails_the_write() {
        let pool = EncodePool::try_new(2, 1).unwrap();
        pool.set_before_encode(Arc::new(|sequence| {
            if sequence == 1 {
                panic!("row group {sequence} failed");
            }
        }));
        let props = props(Some(1_000), None, Arc::clone(&pool));
        let mut queue = queue(&props);
        let mut appended = 0;
        let mut deliver = |_: EncodedRowGroup| {
            appended += 1;
            Ok(())
        };
        let error = queue
            .push(&batch(0, 3_000), &mut deliver)
            .expect_err("a failed row group fails the push");
        assert!(error.to_string().contains("panicked"), "{error}");
        let error = queue
            .finish(&mut deliver)
            .expect_err("a failed queue does not finish");
        assert!(error.to_string().contains("earlier row group"), "{error}");
        assert_eq!(appended, 1, "only the row group before the failed one");
        assert_eq!(pool.encoded_here(), 2);
        assert_eq!(pool.peak_reserved(), 0);
        assert_eq!(pool.out(), 0);
    }

    /// A write dropped while its row groups are still encoding gives their
    /// bytes and counts back once the jobs finish.
    #[test]
    fn a_dropped_write_gives_back_a_shared_pools_budget() {
        let pool = EncodePool::try_new(2, usize::MAX).unwrap();
        let (open, opened) = mpsc::channel::<()>();
        let opened = Mutex::new(opened);
        pool.set_before_encode(Arc::new(move |_| {
            // Every job waits until the gate drops.
            let _ = lock(&opened).recv();
        }));
        let props = props(Some(100), None, Arc::clone(&pool));
        let mut queue = queue(&props);
        // Four row groups, the pool's most out: none waits for another.
        queue.push(&batch(0, 400), &mut |_| Ok(())).unwrap();
        assert_eq!(pool.out(), 4);
        assert!(pool.reserved() > 0);
        drop(queue);
        drop(open);
        wait_until_given_back(&pool);
    }

    #[test]
    fn content_defined_chunking_is_written_on_one_thread() {
        let props = DataFileProperties {
            props: WriterProperties::builder()
                .set_content_defined_chunking(Some(Default::default()))
                .build(),
            dictionary_set: Arc::default(),
            bloom_filter_set: Arc::default(),
            pool: EncodePool::new(2, None),
        };
        assert!(!supports(&props.props));
        assert!(
            RowGroupQueue::try_new(schema(), &props, 1)
                .unwrap()
                .is_none()
        );
        let no_pool = DataFileProperties::from(WriterProperties::builder().build());
        assert!(
            RowGroupQueue::try_new(schema(), &no_pool, 1)
                .unwrap()
                .is_none()
        );
    }
}
