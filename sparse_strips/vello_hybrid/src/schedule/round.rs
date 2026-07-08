// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Concrete round representation for the new hybrid scheduler.

use super::buffer::{Ranges, ScheduleBuffers};
use super::draw::Draw;
use super::pool::Pools;
use super::{LayerTextureRegion, TextureRegion};
use crate::filter::GpuFilterData;
use alloc::vec::Vec;
use core::ops::Range;
use vello_common::geometry::RectU16;
use vello_common::peniko::BlendMode;
use vello_common::util::Clear;

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
    pub(super) fn root_draw_mut(&mut self) -> &mut Draw {
        &mut self.root_draw
    }

    pub(super) fn layer_draw_mut(&mut self, texture_index: usize) -> &mut Draw {
        &mut self.layer_passes[texture_index].draw
    }

    pub(super) fn push_blend(
        &mut self,
        texture_index: usize,
        buffers: &mut ScheduleBuffers,
        blend: BlendOp,
    ) {
        buffers
            .blends
            .push(&mut self.layer_passes[texture_index].blend_ranges, blend);
    }

    pub(super) fn push_filter(
        &mut self,
        texture_index: usize,
        buffers: &mut ScheduleBuffers,
        filter: FilterOp,
    ) {
        buffers
            .filter_ops
            .push(&mut self.layer_passes[texture_index].filter_ranges, filter);
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
pub(super) struct LayerPass {
    pub(super) draw: Draw,
    pub(super) filter_ranges: Ranges,
    pub(super) filter_passes: FilterPasses,
    pub(super) blend_ranges: Ranges,
}

#[derive(Debug, Default)]
pub(crate) struct FilterPasses {
    pub(crate) steps: Vec<Range<usize>>,
    pub(crate) copy_back: Range<usize>,
}

impl FilterPasses {
    pub(crate) fn is_empty(&self) -> bool {
        self.copy_back.is_empty()
    }
}

impl Clear for FilterPasses {
    fn clear(&mut self) {
        self.steps.clear();
        self.copy_back = 0..0;
    }
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
