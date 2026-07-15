// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Monotonic allocation cursor for scheduled rounds.

use crate::schedule::allocate::{
    AllocatedTextureRegion, Allocation, Atlases, LayerAllocationRequest,
};
use crate::target::LayerTextureId;
use alloc::vec::Vec;
use vello_common::multi_atlas::AtlasError;

#[derive(Debug)]
pub(super) struct Cursor {
    current_round: usize,
    atlases: Atlases,
    pending_releases: Vec<Vec<AllocatedTextureRegion<LayerTextureId>>>,
}

impl Cursor {
    pub(super) fn new(atlases: Atlases) -> Self {
        Self {
            current_round: 0,
            atlases,
            pending_releases: Vec::new(),
        }
    }

    pub(super) fn current_round(&self) -> usize {
        self.current_round
    }

    pub(super) fn scratch_texture(&self) -> bool {
        self.atlases.scratch_texture()
    }

    pub(super) fn require_scratch_texture(&mut self) -> Result<(), AtlasError> {
        self.atlases.require_scratch_texture()
    }

    pub(super) fn allocate_layer(
        &mut self,
        request: LayerAllocationRequest,
    ) -> Result<Allocation<AllocatedTextureRegion<LayerTextureId>>, AtlasError> {
        if let Some(allocation) =
            self.allocate_reusing(|atlases| Ok(atlases.allocate_layer(&request)))?
        {
            return Ok(allocation);
        }

        // The currently available layer textures do not have enough room to store our layer.
        // Therefore, we need to create a new one.

        self.atlases.add_layer_atlas(request.texture_parity)?;
        let allocation = self
            .atlases
            .allocate_layer(&request)
            .ok_or(AtlasError::NoSpaceAvailable)?;

        Ok(Allocation {
            allocation,
            round_idx: self.current_round,
        })
    }

    /// Advance the round counter until enough resources have been freed such that
    /// the given allocation succeeds.
    ///
    /// Return `Ok(None)` in case it's not possible to perform the allocation using the
    /// currently available resources.
    fn allocate_reusing<T: Copy>(
        &mut self,
        mut allocate: impl FnMut(&mut Atlases) -> Result<Option<T>, AtlasError>,
    ) -> Result<Option<Allocation<T>>, AtlasError> {
        loop {
            if let Some(allocation) = allocate(&mut self.atlases)? {
                return Ok(Some(Allocation {
                    allocation,
                    round_idx: self.current_round,
                }));
            }

            if self.current_round >= self.pending_releases.len() {
                return Ok(None);
            }

            self.advance();
        }
    }

    pub(super) fn release(
        &mut self,
        allocation: AllocatedTextureRegion<LayerTextureId>,
        round_idx: usize,
    ) {
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
                self.atlases.deallocate(allocation);
            }
        }

        self.current_round += 1;
    }
}
