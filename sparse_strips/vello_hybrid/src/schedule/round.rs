// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Concrete round representation for the new hybrid scheduler.

use super::{Draw, LayerTextureRegion, LoadOp, RenderTarget, ScratchRegion};
use crate::filter::GpuFilterData;
use alloc::vec::Vec;
use vello_common::geometry::RectU16;
use vello_common::peniko::BlendMode;

#[derive(Debug, Default)]
pub(super) struct Rounds {
    pub(super) rounds: Vec<Round>,
}

#[derive(Debug, Default)]
pub(super) struct Round {
    pub(super) root_passes: Vec<RenderPass>,
    pub(super) layer_passes: [LayerPass; 2],
    pub(super) layer_clears: Vec<LayerTextureRegion>,
    pub(super) scratch_clears: Vec<ScratchRegion>,
}

#[derive(Debug, Default)]
pub(super) struct LayerPass {
    pub(super) render_passes: Vec<RenderPass>,
    pub(super) filters: Vec<FilterOp>,
    pub(super) blends: Vec<BlendOp>,
}

#[derive(Debug)]
pub(super) struct RenderPass {
    pub(super) target: RenderTarget,
    pub(super) draw: Draw,
    pub(super) load_op: LoadOp,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FilterOp {
    pub(crate) layer_region: LayerTextureRegion,
    pub(crate) scratches: [Option<ScratchRegion>; 2],
    pub(crate) filter_data_offset: u32,
    pub(crate) gpu_filter: GpuFilterData,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BlendOp {
    pub(crate) parent_region: LayerTextureRegion,
    pub(crate) child_region: LayerTextureRegion,
    pub(crate) blend_bbox: RectU16,
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
        let pass = RenderPass {
            target,
            draw,
            load_op,
        };

        match target {
            RenderTarget::Root(_) => self.root_passes.push(pass),
            RenderTarget::Layer(region) => {
                self.layer_passes[region.texture_index]
                    .render_passes
                    .push(pass);
            }
        }
    }

    pub(super) fn push_blend(&mut self, blend: BlendOp) {
        self.layer_passes[blend.parent_region.texture_index]
            .blends
            .push(blend);
    }

    pub(super) fn push_filter(&mut self, filter: FilterOp) {
        self.layer_passes[filter.layer_region.texture_index]
            .filters
            .push(filter);
    }
}
