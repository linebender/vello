// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Executes planned rendering rounds through a backend.

use super::buffer::Ranges;
use super::draw::Draw;
use super::pool::Pools;
use super::round::{FilterPasses, Rounds};
use super::{
    ExternalTextureRun, RootRenderTarget, Schedule, SchedulePlanner, ScheduleStorage,
    StripPassRenderTarget, TextureRequirements, TextureTarget,
};
use crate::filter::FilterContext;
use crate::paint::PaintResolver;
use crate::{GpuStrip, RenderError, Scene};
use alloc::vec::Vec;
use vello_common::encode::EncodedPaint;
use vello_common::geometry::RectU16;

pub(crate) trait RendererBackend {
    /// Return the persistent storage used while scheduling and executing a scene.
    fn schedule_storage(&mut self) -> &mut ScheduleStorage;

    /// Ensure intermediate layer/scratch textures required by this scene are allocated.
    fn prepare_intermediate_textures(&mut self, requirements: TextureRequirements);

    /// Return the dimensions of each layer atlas texture.
    fn layer_texture_size(&self) -> (u32, u32);

    /// Clear rectangular regions in a texture to transparent black.
    fn clear_rects(&mut self, target: TextureTarget, populate: impl FnOnce(&mut Vec<RectU16>));

    /// Render the global opaque strips to the user-provided root surface.
    fn render_root_opaque(&mut self, strips: &[GpuStrip]);

    /// Render ranged alpha strips to a root or layer target.
    fn render_draw(
        &mut self,
        strips: &Ranges,
        external_texture_runs: &[ExternalTextureRun],
        target: StripPassRenderTarget,
    );

    /// Apply non-default blend layer operations.
    fn blend(&mut self, blends: &Ranges, texture_index: usize);

    /// Apply filter operations to already-rendered layer atlas regions.
    fn apply_filters(&mut self, passes: &FilterPasses, texture_index: usize);
}

/// Render the supported subset of a scene through the recorder-based scheduler.
pub(crate) fn render_scene<R: RendererBackend>(
    renderer: &mut R,
    scene: &Scene,
    root_output_target: RootRenderTarget,
    paint_idxs: &[u32],
    encoded_paints: &[EncodedPaint],
    filter_context: &FilterContext,
) -> Result<(), RenderError> {
    renderer.prepare_intermediate_textures(TextureRequirements::for_scene(scene));
    let strip_storage = scene.strip_storage.borrow();
    let paint_resolver = PaintResolver::new(encoded_paints, paint_idxs);
    let layer_texture_size = renderer.layer_texture_size();
    let schedule = {
        let ScheduleStorage {
            pools,
            buffers,
            filter_plan_scratch,
        } = renderer.schedule_storage();
        buffers.clear();
        let mut planner = SchedulePlanner::new(
            scene,
            &strip_storage,
            root_output_target,
            paint_resolver,
            filter_context,
            layer_texture_size,
            pools,
            buffers,
            filter_plan_scratch,
        );
        planner.build()?
    };
    schedule.execute(renderer, root_output_target);
    let ScheduleStorage { pools, buffers, .. } = renderer.schedule_storage();
    schedule.recycle(pools);
    buffers.clear();
    Ok(())
}

impl Schedule {
    fn execute<R: RendererBackend>(&self, renderer: &mut R, root_output_target: RootRenderTarget) {
        if let Some(strips) = &self.opaque_strips {
            renderer.render_root_opaque(strips);
        }

        self.rounds.execute(renderer, root_output_target);
    }

    fn recycle(self, pools: &mut Pools) {
        pools.submit_opaque_strips(self.opaque_strips);
        self.rounds.recycle(pools);
    }
}

impl Rounds {
    fn execute<R: RendererBackend>(&self, renderer: &mut R, root_output_target: RootRenderTarget) {
        for round in &self.rounds {
            for texture_index in [1, 0] {
                let layer_round = &round.layer_passes[texture_index];
                let draw = &layer_round.draw;
                renderer.render_draw(
                    &draw.strip_ranges,
                    &draw.external_texture_runs,
                    StripPassRenderTarget::LayerAtlas(texture_index),
                );

                renderer.apply_filters(&layer_round.filter_passes, texture_index);
            }

            Self::execute_root_pass(renderer, &round.root_draw, root_output_target);

            for texture_index in 0..round.layer_passes.len() {
                renderer.blend(
                    &round.layer_passes[texture_index].blend_ranges,
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
            renderer.clear_rects(target(texture_index), |clear_rects| {
                clear_rects.extend(regions.iter().copied().filter(|region| !region.is_empty()));
            });
        }
    }

    fn execute_root_pass<R: RendererBackend>(
        renderer: &mut R,
        draw: &Draw,
        root_output_target: RootRenderTarget,
    ) {
        if draw.is_empty() {
            return;
        }

        renderer.render_draw(
            &draw.strip_ranges,
            &draw.external_texture_runs,
            StripPassRenderTarget::Root(root_output_target),
        );
    }
}
