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
    pub(super) debug: ScheduleDebugStats,
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
#[allow(dead_code)]
pub(crate) struct BlendOp {
    pub(crate) parent: LayerTextureRegion,
    pub(crate) source: LayerTextureRegion,
    pub(crate) bbox: RectU16,
    pub(crate) blend_mode: BlendMode,
    pub(crate) opacity: f32,
}

#[derive(Debug, Default)]
pub(super) struct ScheduleDebugStats {
    pub(super) layer_count: usize,
    pub(super) root_cmd_count: usize,
    pub(super) root_child_layer_count: usize,
    pub(super) root_is_blend_target: bool,
    pub(super) layer_texture_size: (u32, u32),
    pub(super) depth_counts: Vec<(usize, usize)>,
    pub(super) allocation_attempts: usize,
    pub(super) allocation_retries: usize,
    pub(super) allocation_retry_events: Vec<AllocationRetryDebug>,
    pub(super) scheduled_layers: Vec<LayerScheduleDebug>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct AllocationRetryDebug {
    pub(super) texture_index: usize,
    pub(super) earliest_round: usize,
    pub(super) allocated_round: usize,
    pub(super) attempts: usize,
    pub(super) bbox: RectU16,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct LayerScheduleDebug {
    pub(super) layer_id: u32,
    pub(super) depth: usize,
    pub(super) texture_index: usize,
    pub(super) command_count: usize,
    pub(super) child_layer_count: usize,
    pub(super) batch_count: usize,
    pub(super) allocated_round: usize,
    pub(super) ready_round: usize,
    pub(super) atlas_x: u32,
    pub(super) atlas_y: u32,
    pub(super) bbox: RectU16,
    pub(super) has_clip: bool,
    pub(super) has_default_blend: bool,
    pub(super) is_destructive_blend: bool,
    pub(super) opacity: f32,
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
