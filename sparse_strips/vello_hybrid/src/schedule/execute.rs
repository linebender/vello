// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Executes planned rendering rounds through a backend.

use super::pool::Pools;
use super::round::{BlendOp, Rounds};
use super::{ExternalTextureRun, Schedule, ScheduleBuffers, ScheduleStorage};
use crate::filter::FilterPassPlan;
use crate::target::{DrawPassTarget, IntermediateTextureSizes, RootRenderTarget, TextureTarget};
use crate::util::{RangedSlice, VecExt};
use crate::{GpuStrip, Scene};
use vello_common::geometry::RectU16;

pub(crate) trait RendererBackend {
    fn opaque_pass(&mut self, strips: &[GpuStrip]);
    fn draw_pass(
        &mut self,
        strips: RangedSlice<'_, GpuStrip>,
        external_texture_runs: &[ExternalTextureRun],
        target: DrawPassTarget,
    );
    fn clear_pass(&mut self, target: TextureTarget, rects: &[RectU16]);
    fn blend_pass(&mut self, blends: RangedSlice<'_, BlendOp>, texture_index: u8);
    fn filter_pass(&mut self, plan: &FilterPassPlan, texture_index: u8);
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
        filter_pass_plan,
    } = storage;
    schedule.execute(renderer, root_output_target, buffers, filter_pass_plan);
    schedule.recycle(pools);
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
            self.texture_sizes,
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
        texture_sizes: IntermediateTextureSizes,
    ) {
        // The core loop that ties everything together!

        // We iterate over each round separately.
        for round in &self.rounds {
            // For each round, we first draws to layer texture 1, then to layer texture 0.
            // The order is important because draws to layer texture 0 might have dependencies
            // on draws to layer texture 1 in the same round!

            for (texture_index, pass) in round.layer_texture_passes.iter().enumerate().rev() {
                let texture_index = u8::try_from(texture_index).unwrap();
                // For each layer texture target, we first perform the draws of all layers that are
                // allocated in this texture.
                let draw = &pass.draw;
                renderer.draw_pass(
                    buffers.strips.ranged(&draw.strip_ranges),
                    &draw.external_texture_runs,
                    DrawPassTarget::Layer(texture_index),
                );

                // Next, we apply all filters for layers in this pass.
                filter_plan.init(
                    buffers
                        .filter_ops
                        .ranged(&pass.filter_ranges)
                        .iter()
                        .copied(),
                    texture_sizes,
                );
                renderer.filter_pass(filter_plan, texture_index);
                // Finally, we apply all blend operations.
                renderer.blend_pass(buffers.blends.ranged(&pass.blend_ranges), texture_index);
            }

            // Once layers are done, we perform any possibly scheduled draws to the root target.
            renderer.draw_pass(
                buffers.strips.ranged(&round.root_draw.strip_ranges),
                &round.root_draw.external_texture_runs,
                DrawPassTarget::Root(root_output_target),
            );

            // Finally, we clear layer regions that are deallocated in this round as well as
            // all painted rectangles in the scratch buffer, so future rounds can assume a clean slate.
            for (texture_index, (layer_clears, scratch_clears)) in round
                .layer_texture_clears
                .iter()
                .zip(round.scratch_texture_clears.iter())
                .enumerate()
            {
                let texture_index = u8::try_from(texture_index).unwrap();

                renderer.clear_pass(TextureTarget::layer(texture_index), layer_clears);
                renderer.clear_pass(TextureTarget::scratch(texture_index), scratch_clears);
            }
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
