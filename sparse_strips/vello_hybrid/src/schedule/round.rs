// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Concrete round representation for the new hybrid scheduler.

use super::{Draw, FilterScratchRegion, LayerTextureRegion, LoadOp, RenderTarget};
use crate::filter::GpuFilterData;
use alloc::vec::Vec;
use vello_common::geometry::RectU16;
use vello_common::peniko::BlendMode;

#[derive(Debug, Default)]
pub(super) struct Schedule {
    pub(super) rounds: Vec<Round>,
}

#[derive(Debug, Default)]
pub(super) struct Round {
    /// Padded filter layer regions that need transparent guard texels before drawing.
    pub(super) prepare_layer_regions: Vec<LayerTextureRegion>,
    /// Padded filter scratch regions that need transparent guard texels before filtering.
    pub(super) prepare_filter_scratch_regions: Vec<FilterScratchRegion>,
    pub(super) passes: Vec<RoundPass>,
    pub(super) filters: [Vec<FilterOp>; 2],
    pub(super) blends: Vec<BlendOp>,
    pub(super) clear_layer_regions: Vec<LayerTextureRegion>,
    pub(super) clear_filter_scratch_regions: Vec<FilterScratchRegion>,
}

#[derive(Debug)]
pub(super) struct RoundPass {
    pub(super) target: RenderTarget,
    pub(super) draw: Draw,
    pub(super) load_op: LoadOp,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FilterOp {
    pub(crate) layer: LayerTextureRegion,
    pub(crate) scratches: [Option<FilterScratchRegion>; 2],
    pub(crate) filter_data_offset: u32,
    pub(crate) gpu_filter: GpuFilterData,
}

#[derive(Debug, Clone, Copy)]
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

    pub(super) fn push_filter(&mut self, filter: FilterOp) {
        self.filters[filter.layer.texture_index].push(filter);
    }
}
