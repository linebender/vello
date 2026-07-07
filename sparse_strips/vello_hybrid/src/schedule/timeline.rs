// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Monotonic resource scheduling for future rounds.

use alloc::vec::Vec;

pub(super) trait ResourceAllocator {
    type Request: Copy;
    type Allocation: Copy;

    fn allocate(&mut self, request: Self::Request) -> Option<Self::Allocation>;

    fn release(&mut self, allocation: Self::Allocation);
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ScheduledAllocation<T> {
    pub(super) allocation: T,
    pub(super) round_idx: usize,
}

#[derive(Debug)]
pub(super) struct Timeline<R: ResourceAllocator> {
    current_round: usize,
    resource: R,
    pending_releases: Vec<Vec<R::Allocation>>,
    pending_release_count: usize,
}

impl<R: ResourceAllocator> Timeline<R> {
    pub(super) fn new(resource: R) -> Self {
        Self {
            current_round: 0,
            resource,
            pending_releases: Vec::new(),
            pending_release_count: 0,
        }
    }

    pub(super) fn base_round(&self) -> usize {
        self.current_round
    }

    pub(super) fn allocate(
        &mut self,
        request: R::Request,
    ) -> Option<ScheduledAllocation<R::Allocation>> {
        loop {
            if let Some(allocation) = self.resource.allocate(request) {
                return Some(ScheduledAllocation {
                    allocation,
                    round_idx: self.current_round,
                });
            }

            if self.pending_release_count == 0 {
                return None;
            }

            self.advance_to(self.current_round + 1);
        }
    }

    pub(super) fn release_after(&mut self, allocation: R::Allocation, round_idx: usize) {
        while self.pending_releases.len() <= round_idx {
            self.pending_releases.push(Vec::new());
        }

        self.pending_releases[round_idx].push(allocation);
        self.pending_release_count += 1;
    }

    fn advance_to(&mut self, round_idx: usize) {
        while self.current_round < round_idx {
            if let Some(releases) = self.pending_releases.get_mut(self.current_round) {
                self.pending_release_count -= releases.len();

                for allocation in releases.drain(..) {
                    self.resource.release(allocation);
                }
            }

            self.current_round += 1;
        }
    }
}
