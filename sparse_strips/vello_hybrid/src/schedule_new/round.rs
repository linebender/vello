// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Concrete round representation for the new hybrid scheduler.

use super::{Draw, LayerTextureRegion, RenderTarget};
use alloc::vec::Vec;

#[derive(Debug, Default)]
pub(super) struct Schedule {
    pub(super) rounds: Vec<Round>,
}

#[derive(Debug, Default)]
pub(super) struct Round {
    pub(super) passes: Vec<RoundPass>,
    pub(super) clear_layer_regions: Vec<LayerTextureRegion>,
}

#[derive(Debug)]
pub(super) struct RoundPass {
    pub(super) target: RenderTarget,
    pub(super) draw: Draw,
}

impl Round {
    pub(super) fn push_pass(&mut self, target: RenderTarget, draw: Draw) {
        self.passes.push(RoundPass { target, draw });
    }
}
