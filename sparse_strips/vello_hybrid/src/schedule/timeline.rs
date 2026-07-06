// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Monotonic resource scheduling for future rounds.

use alloc::vec::Vec;

/// A resource allocator whose allocations can stay live across multiple rounds.
pub(super) trait ResourceAllocator {
    type Request: Copy;
    type Allocation: Copy;

    fn allocate(&mut self, request: Self::Request) -> Option<Self::Allocation>;

    fn release(&mut self, allocation: Self::Allocation);
}

/// Result of a successful monotonic allocation attempt.
#[derive(Debug, Clone, Copy)]
pub(super) struct ScheduledAllocation<T> {
    pub(super) allocation: T,
    pub(super) round_idx: usize,
}

/// A small online scheduler for resources whose state only moves forward in time.
#[derive(Debug)]
pub(super) struct Timeline<R: ResourceAllocator> {
    base_round: usize,
    resource: R,
    pending_releases: Vec<Vec<R::Allocation>>,
}

impl<R: ResourceAllocator> Timeline<R> {
    pub(super) fn new(resource: R) -> Self {
        Self {
            base_round: 0,
            resource,
            pending_releases: Vec::new(),
        }
    }

    pub(super) fn base_round(&self) -> usize {
        self.base_round
    }

    /// Try to allocate at the current base round.
    ///
    /// If the resource is full, the scheduler advances round-by-round and applies releases that
    /// were scheduled after completed rounds. Pressure creates more rounds, but the scheduler
    /// never patches historical states.
    pub(super) fn allocate<F>(
        &mut self,
        request: R::Request,
        mut ensure_round_exists: F,
    ) -> Option<ScheduledAllocation<R::Allocation>>
    where
        F: FnMut(usize),
    {
        loop {
            ensure_round_exists(self.base_round);
            if let Some(allocation) = self.resource.allocate(request) {
                return Some(ScheduledAllocation {
                    allocation,
                    round_idx: self.base_round,
                });
            }

            if !self.has_pending_release() {
                return None;
            }

            self.advance_to_round(self.base_round + 1, &mut ensure_round_exists);
        }
    }

    pub(super) fn release_after<F>(
        &mut self,
        allocation: R::Allocation,
        round_idx: usize,
        mut ensure_round_exists: F,
    ) where
        F: FnMut(usize),
    {
        ensure_round_exists(round_idx);
        while self.pending_releases.len() <= round_idx {
            self.pending_releases.push(Vec::new());
        }
        self.pending_releases[round_idx].push(allocation);
    }

    fn advance_to_round<F>(&mut self, round_idx: usize, ensure_round_exists: &mut F)
    where
        F: FnMut(usize),
    {
        ensure_round_exists(round_idx);
        while self.base_round < round_idx {
            if let Some(releases) = self.pending_releases.get_mut(self.base_round) {
                for allocation in releases.drain(..) {
                    self.resource.release(allocation);
                }
            }
            self.base_round += 1;
        }
    }

    fn has_pending_release(&self) -> bool {
        self.pending_releases
            .iter()
            .skip(self.base_round)
            .any(|releases| !releases.is_empty())
    }
}
