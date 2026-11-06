//! A counting global allocator for the crate's unit tests and its
//! integration-test binary (which includes this file as a module), and a
//! memory pool that publishes its reservation to it, so a test can check that
//! the memory a piece of code allocates never exceeds what it has reserved.
//!
//! Counting is per thread and off unless a test turns it on with [`measure`],
//! so the other tests, which run on other threads, are not counted.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use datafusion::common::Result;
use datafusion::execution::memory_pool::{MemoryPool, MemoryReservation};

struct Counting;

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    /// Bytes allocated and not yet freed since counting started.
    static LIVE: Cell<isize> = const { Cell::new(0) };
    /// The bytes a [`BudgetPool`] has reserved on this thread.
    static RESERVED: Cell<usize> = const { Cell::new(0) };
    /// The largest `live - reserved` seen at an allocation.
    static EXCESS: Cell<isize> = const { Cell::new(isize::MIN) };
    static PEAK: Cell<isize> = const { Cell::new(0) };
}

fn counting() -> bool {
    ENABLED.try_with(Cell::get).unwrap_or(false)
}

/// Record `bytes` becoming live.
fn allocated(bytes: usize) {
    let _ = LIVE.try_with(|live| {
        let now = live.get() + bytes as isize;
        let reserved = RESERVED.with(Cell::get) as isize;
        EXCESS.with(|e| e.set(e.get().max(now - reserved)));
        PEAK.with(|p| p.set(p.get().max(now)));
        live.set(now);
    });
}

fn freed(bytes: usize) {
    let _ = LIVE.try_with(|live| live.set(live.get() - bytes as isize));
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if counting() {
            allocated(layout.size());
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if counting() {
            allocated(layout.size());
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if counting() {
            freed(layout.size());
        }
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if counting() {
            // Counted as a copying reallocation: the new block is allocated
            // while the old one, already counted, is still live.
            allocated(new_size);
            freed(layout.size());
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// What [`measure`] saw while its closure ran on this thread.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Measured {
    /// The largest number of bytes live at once.
    pub(crate) peak: usize,
    /// The largest amount by which the live bytes exceeded the reservation
    /// of the thread's [`BudgetPool`] at any allocation; at most 0 when every
    /// allocation was reserved beforehand.
    pub(crate) excess: isize,
}

/// Run `f` with this thread's allocations counted.
pub(crate) fn measure<T>(f: impl FnOnce() -> T) -> (T, Measured) {
    LIVE.with(|l| l.set(0));
    PEAK.with(|p| p.set(0));
    EXCESS.with(|e| e.set(isize::MIN));
    ENABLED.with(|e| e.set(true));
    let out = f();
    ENABLED.with(|e| e.set(false));
    let measured = Measured {
        peak: PEAK.with(Cell::get).max(0) as usize,
        excess: EXCESS.with(Cell::get),
    };
    (out, measured)
}

/// An unbounded memory pool that publishes its reservation to this thread's
/// counter and remembers its peak.
#[derive(Debug, Default)]
pub(crate) struct BudgetPool {
    reserved: AtomicUsize,
    peak: AtomicUsize,
}

impl BudgetPool {
    pub(crate) fn peak(&self) -> usize {
        self.peak.load(Ordering::SeqCst)
    }

    fn publish(&self) {
        let reserved = self.reserved.load(Ordering::SeqCst);
        self.peak.fetch_max(reserved, Ordering::SeqCst);
        let _ = RESERVED.try_with(|r| r.set(reserved));
    }
}

impl std::fmt::Display for BudgetPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BudgetPool")
    }
}

impl MemoryPool for BudgetPool {
    fn name(&self) -> &str {
        "BudgetPool"
    }

    fn grow(&self, _reservation: &MemoryReservation, additional: usize) {
        self.reserved.fetch_add(additional, Ordering::SeqCst);
        self.publish();
    }

    fn shrink(&self, _reservation: &MemoryReservation, shrink: usize) {
        self.reserved.fetch_sub(shrink, Ordering::SeqCst);
        self.publish();
    }

    fn try_grow(&self, reservation: &MemoryReservation, additional: usize) -> Result<()> {
        self.grow(reservation, additional);
        Ok(())
    }

    fn reserved(&self) -> usize {
        self.reserved.load(Ordering::SeqCst)
    }
}
