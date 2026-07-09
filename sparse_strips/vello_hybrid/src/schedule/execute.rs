// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Executes planned rendering rounds through a backend.

use super::buffer::{RangedSlice, ScheduleBuffers, VecExt};
use super::pool::Pools;
use super::round::{BlendOp, Rounds};
use super::{
    ExternalTextureRun, RootRenderTarget, Schedule, ScheduleStorage, StripPassRenderTarget,
    TextureTarget,
};
use crate::filter::{FilterPassPlan, build_plan};
use crate::{GpuStrip, Scene};
use alloc::vec::Vec;
use vello_common::geometry::RectU16;

pub(crate) trait RendererBackend {
    /// Clear rectangular regions in a texture to transparent black.
    fn clear_pass(&mut self, target: TextureTarget, rects: &[RectU16]);

    /// Render the global opaque strips to the user-provided root surface.
    fn opaque_pass(&mut self, strips: &[GpuStrip]);

    /// Render ranged alpha strips to a root or layer target.
    fn draw_pass(
        &mut self,
        strips: RangedSlice<'_, GpuStrip>,
        external_texture_runs: &[ExternalTextureRun],
        target: StripPassRenderTarget,
    );

    /// Apply non-default blend layer operations.
    fn blend_pass(&mut self, blends: RangedSlice<'_, BlendOp>, texture_index: usize);

    /// Apply filter operations to already-rendered layer atlas regions.
    fn filter_pass(&mut self, plan: &FilterPassPlan, texture_index: usize);
}

pub(crate) fn execute<R: RendererBackend>(
    renderer: &mut R,
    storage: &mut ScheduleStorage,
    schedule: Schedule,
    root_output_target: RootRenderTarget,
) {
    let ScheduleStorage {
        pools,
        buffers,
        filter_pass_plan: filter_plan,
    } = storage;
    schedule.execute(renderer, root_output_target, buffers, filter_plan);
    schedule.recycle(pools);
    buffers.clear();
}

impl Schedule {
    fn execute<R: RendererBackend>(
        &self,
        renderer: &mut R,
        root_output_target: RootRenderTarget,
        buffers: &ScheduleBuffers,
        filter_plan: &mut FilterPassPlan,
    ) {
        if let Some(strips) = &self.opaque_strips {
            renderer.opaque_pass(strips);
        }

        self.rounds.execute(
            renderer,
            root_output_target,
            buffers,
            filter_plan,
            self.layer_texture_size,
        );
    }

    fn recycle(self, pools: &mut Pools) {
        pools.submit_opaque_strips(self.opaque_strips);
        self.rounds.recycle(pools);
    }
}

impl Rounds {
    fn execute<R: RendererBackend>(
        &self,
        renderer: &mut R,
        root_output_target: RootRenderTarget,
        buffers: &ScheduleBuffers,
        filter_plan: &mut FilterPassPlan,
        layer_texture_size: (u16, u16),
    ) {
        for round in &self.rounds {
            for (index, pass) in round.layer_texture_passes.iter().enumerate().rev() {
                let draw = &pass.draw;
                renderer.draw_pass(
                    buffers.strips.ranged(&draw.strip_ranges),
                    &draw.external_texture_runs,
                    StripPassRenderTarget::LayerAtlas(index),
                );

                let filter_ops = buffers.filter_ops.ranged(&pass.filter_ranges);
                build_plan(filter_ops.iter().copied(), layer_texture_size, filter_plan);
                renderer.filter_pass(filter_plan, index);
                renderer.blend_pass(buffers.blends.ranged(&pass.blend_ranges), index);
            }

            renderer.draw_pass(
                buffers.strips.ranged(&round.root_draw.strip_ranges),
                &round.root_draw.external_texture_runs,
                StripPassRenderTarget::Root(root_output_target),
            );

            Self::clear_regions(renderer, &round.layer_texture_clears, TextureTarget::layer);
            Self::clear_regions(
                renderer,
                &round.scratch_texture_clears,
                TextureTarget::scratch,
            );
        }
    }

    fn clear_regions<R: RendererBackend>(
        renderer: &mut R,
        regions: &[Vec<RectU16>; 2],
        target: impl Fn(usize) -> TextureTarget,
    ) {
        for (texture_index, regions) in regions.iter().enumerate() {
            renderer.clear_pass(target(texture_index), regions);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TextureRequirements {
    pub(crate) layer_textures: [bool; 2],
    pub(crate) scratch_textures: [bool; 2],
}

impl TextureRequirements {
    pub(crate) fn new(scene: &Scene) -> Self {
        let mut layer_textures = [false; 2];
        if scene.recorder.root_is_blend_target {
            // When the root is blended into, the root as a whole needs to be rendered as a layer
            // first so it can be sampled from, and then blit back into the user-provided view
            // in the very end.
            layer_textures[1] = true;
        }

        // Determine how many layer textures we need based on the maximum layer depth, taking our
        // ping-pong scheme into consideration.
        let depth_offset = usize::from(scene.recorder.root_is_blend_target);
        for depth in 1..=scene.recorder.max_layer_depth.min(2) {
            layer_textures[(depth + depth_offset) & 1] = true;
        }

        // Filter layers need 2 textures for ping-ponging, for blending we only need the first one.
        let scratch_textures = if scene.recorder.has_filter_layer {
            [true, true]
        } else if scene.recorder.has_non_default_blend {
            [true, false]
        } else {
            [false, false]
        };

        Self {
            layer_textures,
            scratch_textures,
        }
    }
}
