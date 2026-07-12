// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Monotonic allocation cursor for scheduled rounds.

use crate::schedule::allocate::{Allocation, Allocator};
use alloc::vec::Vec;
use vello_common::multi_atlas::AtlasError;

#[derive(Debug)]
pub(super) struct Cursor<R: Allocator> {
    current_round: usize,
    resource: R,
    pending_releases: Vec<Vec<R::Release>>,
}

impl<R: Allocator> Cursor<R> {
    pub(super) fn new(resource: R) -> Self {
        Self {
            current_round: 0,
            resource,
            pending_releases: Vec::new(),
        }
    }

    pub(super) fn current_round(&self) -> usize {
        self.current_round
    }

    pub(super) fn allocate(
        &mut self,
        request: R::Request,
    ) -> Result<Allocation<R::Allocation>, AtlasError> {
        loop {
            if let Some(allocation) = self.resource.allocate(request) {
                return Ok(Allocation {
                    allocation,
                    round_idx: self.current_round,
                });
            }

            if self.current_round >= self.pending_releases.len() {
                return Err(AtlasError::NoSpaceAvailable);
            }

            self.advance();
        }
    }

    pub(super) fn release(&mut self, allocation: R::Release, round_idx: usize) {
        assert!(
            round_idx >= self.current_round,
            "cannot release an allocation in a round already passed by the cursor"
        );

        while self.pending_releases.len() <= round_idx {
            self.pending_releases.push(Vec::new());
        }

        self.pending_releases[round_idx].push(allocation);
    }

    /// Advance to the next round.
    fn advance(&mut self) {
        if let Some(releases) = self.pending_releases.get_mut(self.current_round) {
            for allocation in releases.drain(..) {
                self.resource.release(allocation);
            }
        }

        self.current_round += 1;
    }
}
