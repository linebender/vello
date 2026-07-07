// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Concrete round representation for the new hybrid scheduler.

use super::{Draw, LayerTextureRegion, TextureRegion};
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
    pub(super) root_pass: RootPass,
    pub(super) layer_passes: [LayerPass; 2],
    pub(super) layer_texture_clears: Vec<LayerTextureRegion>,
    pub(super) scratch_texture_clears: Vec<TextureRegion>,
}

impl Round {
    pub(crate) fn push_root_draw(&mut self, draw: Draw) {
        self.root_pass.draw.append(&draw);
    }

    pub(crate) fn push_layer_draw(&mut self, texture_index: usize, draw: Draw) {
        self.layer_passes[texture_index].draw.append(&draw);
    }

    pub(crate) fn push_blend(&mut self, blend: BlendOp) {
        self.layer_passes[blend.parent_region.texture.texture_index]
            .blends
            .push(blend);
    }

    pub(crate) fn push_filter(&mut self, filter: FilterOp) {
        self.layer_passes[filter.layer_region.texture.texture_index]
            .filters
            .push(filter);
    }
}

#[derive(Debug, Default)]
pub(super) struct LayerPass {
    pub(super) draw: Draw,
    pub(super) filters: Vec<FilterOp>,
    pub(super) blends: Vec<BlendOp>,
}

#[derive(Debug, Default)]
pub(super) struct RootPass {
    pub(super) draw: Draw,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FilterOp {
    pub(crate) layer_region: LayerTextureRegion,
    pub(crate) scratches: [Option<TextureRegion>; 2],
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
