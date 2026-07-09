// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Executes planned rendering rounds through a backend.

use super::buffer::{RangedSlice, ScheduleBuffers, VecExt};
use super::pool::Pools;
use super::round::{BlendOp, ResolvedFilterPasses, Rounds};
use super::{
    ExternalTextureRun, RootRenderTarget, Schedule, SchedulePlanner, ScheduleStorage,
    StripPassRenderTarget, TextureTarget,
};
use crate::filter::FilterContext;
use crate::paint::PaintResolver;
use crate::{GpuStrip, RenderError, Scene};
use alloc::vec::Vec;
use vello_common::geometry::RectU16;

pub(crate) trait RendererBackend {
    /// Return the dimensions of each layer atlas texture.
    fn layer_texture_size(&self) -> (u32, u32);

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
    fn filter_pass(&mut self, passes: ResolvedFilterPasses<'_>, texture_index: usize);
}

/// Render the supported subset of a scene through the recorder-based scheduler.
pub(crate) fn execute<R: RendererBackend>(
    renderer: &mut R,
    storage: &mut ScheduleStorage,
    scene: &Scene,
    root_output_target: RootRenderTarget,
    paint_resolver: PaintResolver<'_>,
    filter_context: &FilterContext,
) -> Result<(), RenderError> {
    let strip_storage = scene.strip_storage.borrow();
    let layer_texture_size = renderer.layer_texture_size();
    storage.buffers.clear();
    let schedule = SchedulePlanner::new(
        scene,
        &strip_storage,
        root_output_target,
        paint_resolver,
        filter_context,
        layer_texture_size,
        storage,
    )
    .build()?;
    schedule.execute(renderer, root_output_target, &storage.buffers);
    schedule.recycle(&mut storage.pools);
    storage.buffers.clear();
    Ok(())
}

impl Schedule {
    fn execute<R: RendererBackend>(
        &self,
        renderer: &mut R,
        root_output_target: RootRenderTarget,
        buffers: &ScheduleBuffers,
    ) {
        if let Some(strips) = &self.opaque_strips {
            renderer.opaque_pass(strips);
        }

        self.rounds.execute(renderer, root_output_target, buffers);
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
    ) {
        for round in &self.rounds {
            for texture_index in [1, 0] {
                let layer_texture_pass = &round.layer_texture_passes[texture_index];
                let draw = &layer_texture_pass.draw;
                renderer.draw_pass(
                    buffers.strips.ranged(&draw.strip_ranges),
                    &draw.external_texture_runs,
                    StripPassRenderTarget::LayerAtlas(texture_index),
                );

                renderer.filter_pass(
                    layer_texture_pass.filter_passes.resolve(buffers),
                    texture_index,
                );
            }

            renderer.draw_pass(
                buffers.strips.ranged(&round.root_draw.strip_ranges),
                &round.root_draw.external_texture_runs,
                StripPassRenderTarget::Root(root_output_target),
            );

            for texture_index in 0..round.layer_texture_passes.len() {
                renderer.blend_pass(
                    buffers
                        .blends
                        .ranged(&round.layer_texture_passes[texture_index].blend_ranges),
                    texture_index,
                );
            }
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
