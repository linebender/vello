// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Concrete round representation for the new hybrid scheduler.

use super::ScheduleBuffers;
use crate::draw::Draw;
use crate::filter::GpuFilterData;
use crate::target::{LayerTextureRegion, TextureIndex, TextureRegion};
use crate::util::{Ranges, VecExt};
use alloc::vec::Vec;
use vello_common::geometry::RectU16;
use vello_common::peniko::BlendMode;

// Note that the field order of these enums matters since we implement
// `PartialOrd` and use this to represent the order in which the stages
// happen.

/// A stage in the execution of a rendering round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum RoundStage {
    Start,
    Even(LayerStage),
    Odd(LayerStage),
    RootDraw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum LayerStage {
    Draw,
    Filter,
    Blend,
}

impl RoundStage {
    pub(super) const fn draw(texture_index: TextureIndex) -> Self {
        match texture_index {
            TextureIndex::Even => Self::Even(LayerStage::Draw),
            TextureIndex::Odd => Self::Odd(LayerStage::Draw),
        }
    }

    pub(super) const fn filter(texture_index: TextureIndex) -> Self {
        match texture_index {
            TextureIndex::Even => Self::Even(LayerStage::Filter),
            TextureIndex::Odd => Self::Odd(LayerStage::Filter),
        }
    }

    pub(super) const fn blend(texture_index: TextureIndex) -> Self {
        match texture_index {
            TextureIndex::Even => Self::Even(LayerStage::Blend),
            TextureIndex::Odd => Self::Odd(LayerStage::Blend),
        }
    }
}

// As for `RoundStage`, the order of fields here is important!
/// A precise point in the execution timeline of the schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct SchedulePoint {
    pub(super) round: usize,
    pub(super) stage: RoundStage,
}

impl SchedulePoint {
    pub(super) const fn start(round: usize) -> Self {
        Self {
            round,
            stage: RoundStage::Start,
        }
    }

    /// Return the first occurrence of `stage` strictly after this point.
    pub(super) fn next(self, stage: RoundStage) -> Self {
        if stage > self.stage {
            Self {
                round: self.round,
                stage,
            }
        } else {
            Self {
                round: self.round + 1,
                stage,
            }
        }
    }
}

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

    pub(super) fn layer_draw_mut(&mut self, texture_index: TextureIndex) -> &mut Draw {
        &mut self.layer_texture_passes[texture_index.get_index()].draw
    }

    pub(super) fn push_blend_op(
        &mut self,
        parent_texture_index: TextureIndex,
        buffers: &mut ScheduleBuffers,
        blend: BlendOp,
    ) {
        buffers.blend_ops.push_ranged(
            &mut self.layer_texture_passes[parent_texture_index.get_index()].blend_ranges,
            blend,
        );
    }

    pub(super) fn push_filter_op(
        &mut self,
        texture_index: TextureIndex,
        buffers: &mut ScheduleBuffers,
        filter: FilterOp,
    ) {
        buffers.filter_ops.push_ranged(
            &mut self.layer_texture_passes[texture_index.get_index()].filter_ranges,
            filter,
        );
    }
}

impl Rounds {
    pub(super) fn ensure_exists(&mut self, round_idx: usize) {
        while self.rounds.len() <= round_idx {
            self.rounds.push(Round::default());
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
