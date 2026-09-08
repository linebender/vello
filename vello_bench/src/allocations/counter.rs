// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::alloc::{GlobalAlloc, Layout, System};
use std::cmp::Ordering;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

static ACTIVE: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static REALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);

/// A system allocator wrapper that records allocations inside [`measure`].
#[derive(Debug)]
pub struct CountingAllocator;

// SAFETY: Every operation delegates to `System` with the original arguments. The additional
// bookkeeping does not inspect or modify allocated memory.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller upholds `GlobalAlloc::alloc`'s contract.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            record_allocation(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller upholds `GlobalAlloc::alloc_zeroed`'s contract.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            record_allocation(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size(), Relaxed);
        // SAFETY: The caller upholds `GlobalAlloc::dealloc`'s contract.
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: The caller upholds `GlobalAlloc::realloc`'s contract.
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            let old_size = layout.size();
            if new_size >= old_size {
                let added = new_size - old_size;
                let live = LIVE_BYTES.fetch_add(added, Relaxed) + added;
                record_peak(live);
            } else {
                LIVE_BYTES.fetch_sub(old_size - new_size, Relaxed);
            }

            if ACTIVE.load(Relaxed) {
                REALLOCATIONS.fetch_add(1, Relaxed);
                ALLOCATED_BYTES.fetch_add(new_size.saturating_sub(old_size), Relaxed);
            }
        }
        new_ptr
    }
}

/// Allocation activity observed while running one measured operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocationStats {
    /// Successful allocation calls.
    pub allocations: usize,
    /// Successful reallocation calls.
    pub reallocations: usize,
    /// Bytes newly requested from allocation and growing reallocation calls.
    pub allocated_bytes: usize,
    /// Maximum increase in live requested bytes over the starting value.
    pub additional_peak_bytes: usize,
    /// Change in live requested bytes from the beginning to the end.
    pub retained_bytes: isize,
}

/// Measures allocations made while executing `operation`.
///
/// Only one measurement may be active in the process. The measured operation should run on one
/// thread and avoid unrelated background work because the allocator observes all process threads.
pub fn measure<T>(operation: impl FnOnce() -> T) -> (T, AllocationStats) {
    assert!(
        ACTIVE
            .compare_exchange(false, true, Relaxed, Relaxed)
            .is_ok(),
        "allocation measurements cannot be nested"
    );

    ALLOCATIONS.store(0, Relaxed);
    REALLOCATIONS.store(0, Relaxed);
    ALLOCATED_BYTES.store(0, Relaxed);
    let initial_live_bytes = LIVE_BYTES.load(Relaxed);
    PEAK_LIVE_BYTES.store(initial_live_bytes, Relaxed);

    let mut guard = MeasurementGuard(true);
    let output = operation();
    ACTIVE.store(false, Relaxed);
    guard.0 = false;

    let final_live_bytes = LIVE_BYTES.load(Relaxed);
    let peak_live_bytes = PEAK_LIVE_BYTES.load(Relaxed);
    let stats = AllocationStats {
        allocations: ALLOCATIONS.load(Relaxed),
        reallocations: REALLOCATIONS.load(Relaxed),
        allocated_bytes: ALLOCATED_BYTES.load(Relaxed),
        additional_peak_bytes: peak_live_bytes.saturating_sub(initial_live_bytes),
        retained_bytes: signed_difference(final_live_bytes, initial_live_bytes),
    };

    (output, stats)
}

fn record_allocation(size: usize) {
    let live = LIVE_BYTES.fetch_add(size, Relaxed) + size;
    if ACTIVE.load(Relaxed) {
        ALLOCATIONS.fetch_add(1, Relaxed);
        ALLOCATED_BYTES.fetch_add(size, Relaxed);
        record_peak(live);
    }
}

fn record_peak(live: usize) {
    if ACTIVE.load(Relaxed) {
        PEAK_LIVE_BYTES.fetch_max(live, Relaxed);
    }
}

fn signed_difference(value: usize, baseline: usize) -> isize {
    match value.cmp(&baseline) {
        Ordering::Greater => isize::try_from(value - baseline).unwrap_or(isize::MAX),
        Ordering::Equal => 0,
        Ordering::Less => -isize::try_from(baseline - value).unwrap_or(isize::MAX),
    }
}

struct MeasurementGuard(bool);

impl Drop for MeasurementGuard {
    fn drop(&mut self) {
        if self.0 {
            ACTIVE.store(false, Relaxed);
        }
    }
}
