// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Concrete round representation for the new hybrid scheduler.

use super::{Draw, LayerTextureRegion, LoadOp, RenderTarget};
use alloc::vec::Vec;
use vello_common::geometry::RectU16;
use vello_common::peniko::BlendMode;

#[derive(Debug, Default)]
pub(super) struct Schedule {
    pub(super) rounds: Vec<Round>,
}

#[derive(Debug, Default)]
pub(super) struct Round {
    pub(super) passes: Vec<RoundPass>,
    pub(super) blends: Vec<BlendOp>,
    pub(super) clear_layer_regions: Vec<LayerTextureRegion>,
}

#[derive(Debug)]
pub(super) struct RoundPass {
    pub(super) target: RenderTarget,
    pub(super) draw: Draw,
    pub(super) load_op: LoadOp,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(all(target_arch = "wasm32", feature = "webgl"), allow(dead_code))]
pub(crate) struct BlendOp {
    pub(crate) parent: LayerTextureRegion,
    pub(crate) source: LayerTextureRegion,
    pub(crate) bbox: RectU16,
    pub(crate) blend_mode: BlendMode,
    pub(crate) opacity: f32,
}

impl Round {
    pub(super) fn push_pass(&mut self, target: RenderTarget, draw: Draw) {
        self.push_pass_with_load(target, draw, LoadOp::Load);
    }

    pub(super) fn push_pass_with_load(
        &mut self,
        target: RenderTarget,
        draw: Draw,
        load_op: LoadOp,
    ) {
        self.passes.push(RoundPass {
            target,
            draw,
            load_op,
        });
    }

    pub(super) fn push_blend(&mut self, blend: BlendOp) {
        self.blends.push(blend);
    }
}
