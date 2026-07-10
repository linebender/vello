// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Concrete round representation for the new hybrid scheduler.

use super::draw::Draw;
use super::pool::Pools;
use super::{LayerTextureRegion, ScheduleBuffers, TextureRegion};
use crate::filter::GpuFilterData;
use crate::util::{Ranges, VecExt};
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
    pub(super) layer_texture_passes: [LayerTexturePass; 2],
    pub(super) layer_texture_clears: [Vec<RectU16>; 2],
    pub(super) scratch_texture_clears: [Vec<RectU16>; 2],
}

impl Round {
    pub(super) fn root_draw_mut(&mut self) -> &mut Draw {
        &mut self.root_draw
    }

    pub(super) fn layer_draw_mut(&mut self, texture_index: usize) -> &mut Draw {
        &mut self.layer_texture_passes[texture_index].draw
    }

    pub(super) fn push_blend_op(
        &mut self,
        parent_texture_index: usize,
        buffers: &mut ScheduleBuffers,
        blend: BlendOp,
    ) {
        buffers.blends.push_ranged(
            &mut self.layer_texture_passes[parent_texture_index].blend_ranges,
            blend,
        );
    }

    pub(super) fn push_filter_op(
        &mut self,
        texture_index: usize,
        buffers: &mut ScheduleBuffers,
        filter: FilterOp,
    ) {
        buffers.filter_ops.push_ranged(
            &mut self.layer_texture_passes[texture_index].filter_ranges,
            filter,
        );
    }
}

impl Rounds {
    pub(super) fn ensure_exists(&mut self, round_idx: usize, pools: &mut Pools) {
        while self.rounds.len() <= round_idx {
            self.rounds.push(pools.take_round());
        }
    }

    pub(super) fn recycle(self, pools: &mut Pools) {
        for round in self.rounds {
            pools.submit_round(round);
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct LayerTexturePass {
    pub(super) draw: Draw,
    pub(super) filter_ranges: Ranges,
    pub(super) blend_ranges: Ranges,
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
