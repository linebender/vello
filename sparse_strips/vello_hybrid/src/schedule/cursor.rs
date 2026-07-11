// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Monotonic allocation cursor for scheduled rounds.

use crate::schedule::allocate::{Allocation, Allocator};
use alloc::vec::Vec;

#[derive(Debug)]
pub(super) struct Cursor<R: Allocator> {
    current_round: usize,
    resource: R,
    pending_releases: Vec<Vec<R::Allocation>>,
    pending_release_count: usize,
}

impl<R: Allocator> Cursor<R> {
    pub(super) fn new(resource: R) -> Self {
        Self {
            current_round: 0,
            resource,
            pending_releases: Vec::new(),
            pending_release_count: 0,
        }
    }

    pub(super) fn current_round(&self) -> usize {
        self.current_round
    }

    pub(super) fn allocate(&mut self, request: R::Request) -> Option<Allocation<R::Allocation>> {
        loop {
            if let Some(allocation) = self.resource.allocate(request) {
                return Some(Allocation {
                    allocation,
                    round_idx: self.current_round,
                });
            }

            if self.pending_release_count == 0 {
                return None;
            }

            self.advance();
        }
    }

    pub(super) fn release(&mut self, allocation: R::Allocation, round_idx: usize) {
        while self.pending_releases.len() <= round_idx {
            self.pending_releases.push(Vec::new());
        }

        self.pending_releases[round_idx].push(allocation);
        self.pending_release_count += 1;
    }

    /// Advance to the next round.
    fn advance(&mut self) {
        if let Some(releases) = self.pending_releases.get_mut(self.current_round) {
            self.pending_release_count -= releases.len();

            for allocation in releases.drain(..) {
                self.resource.release(allocation);
            }
        }

        self.current_round += 1;
    }
}
