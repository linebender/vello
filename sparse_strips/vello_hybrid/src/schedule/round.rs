// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Concrete round representation for the new hybrid scheduler.

use super::draw::Draw;
use super::{LayerTextureRegion, TextureRegion};
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
    pub(super) root_draw: Draw,
    pub(super) layer_passes: [LayerPass; 2],
    pub(super) layer_texture_clears: [Vec<RectU16>; 2],
    pub(super) scratch_texture_clears: [Vec<RectU16>; 2],
}

impl Round {
    pub(crate) fn root_draw_mut(&mut self) -> &mut Draw {
        &mut self.root_draw
    }

    pub(crate) fn layer_draw_mut(&mut self, texture_index: usize) -> &mut Draw {
        &mut self.layer_passes[texture_index].draw
    }

    pub(crate) fn layer_blends_mut(&mut self, texture_index: usize) -> &mut Vec<BlendOp> {
        &mut self.layer_passes[texture_index].blends
    }

    pub(crate) fn layer_filters_mut(&mut self, texture_index: usize) -> &mut Vec<FilterOp> {
        &mut self.layer_passes[texture_index].filters
    }
}

#[derive(Debug, Default)]
pub(super) struct LayerPass {
    pub(super) draw: Draw,
    pub(super) filters: Vec<FilterOp>,
    pub(super) blends: Vec<BlendOp>,
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
