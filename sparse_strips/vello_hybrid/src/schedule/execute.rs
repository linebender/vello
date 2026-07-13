// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Executes planned rendering rounds through a backend.

use super::round::{BlendOp, Rounds};
use super::{Schedule, ScheduleBuffers, ScheduleStorage};
use crate::GpuStrip;
use crate::draw::ExternalTextureRun;
use crate::filter::FilterPassPlan;
use crate::target::{
    DrawPassTarget, IntermediateTextureSizes, LayerTextureId, LayerTexturePair, RootRenderTarget,
    TextureParity, TextureTarget,
};
use crate::util::{RangedSlice, VecExt};
use vello_common::geometry::RectU16;

pub(crate) trait RendererBackend {
    fn opaque_pass(&mut self, strips: &[GpuStrip]);
    fn draw_pass(
        &mut self,
        strips: RangedSlice<'_, GpuStrip>,
        external_texture_runs: &[ExternalTextureRun],
        target: DrawPassTarget,
        child_layer_texture: Option<LayerTextureId>,
    );
    fn clear_pass(&mut self, target: TextureTarget, rects: &[RectU16]);
    fn blend_pass(
        &mut self,
        blends: RangedSlice<'_, BlendOp>,
        parent_texture_parity: TextureParity,
        texture_pair: LayerTexturePair,
    );
    fn filter_pass(&mut self, plan: &FilterPassPlan, layer_id: LayerTextureId);
}

pub(crate) fn execute<R: RendererBackend>(
    renderer: &mut R,
    storage: &mut ScheduleStorage,
    schedule: Schedule,
    root_output_target: RootRenderTarget,
) {
    let ScheduleStorage {
        buffers,
        filter_pass_plan,
        ..
    } = storage;
    schedule.execute(renderer, root_output_target, buffers, filter_pass_plan);
}

impl Schedule {
    fn execute<R: RendererBackend>(
        &self,
        renderer: &mut R,
        root_output_target: RootRenderTarget,
        buffers: &ScheduleBuffers,
        filter_plan: &mut FilterPassPlan,
    ) {
        if DrawPassTarget::Root(root_output_target).enable_opaque() {
            renderer.opaque_pass(&buffers.draw_buffers.opaque_strips);
        }

        self.rounds.execute(
            renderer,
            root_output_target,
            buffers,
            filter_plan,
            self.texture_sizes,
        );
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
            let texture_pair = round.texture_pair();
            // For each round, we first draw to the even layer texture, then to the odd layer
            // texture. The order is important because odd layers can depend on draws to the even
            // texture in the same round.

            for (index, pass) in round.layer_texture_passes.iter().enumerate() {
                let texture_parity = TextureParity::from_parity(index);
                // For each layer texture target, we first perform the draws of all layers that are
                // allocated in this texture.
                let draw = &pass.draw;
                let layer_id = texture_pair.layer_id(texture_parity);
                renderer.draw_pass(
                    buffers.draw_buffers.strips.ranged(&draw.strip_ranges),
                    &draw.external_texture_runs,
                    DrawPassTarget::Layer(layer_id),
                    draw.has_child_layer
                        .then(|| texture_pair.layer_id(texture_parity.opposite())),
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
                renderer.filter_pass(filter_plan, layer_id);
                // Finally, we apply all blend operations.
                renderer.blend_pass(
                    buffers.blend_ops.ranged(&pass.blend_ranges),
                    texture_parity,
                    texture_pair,
                );
            }

            // Once layers are done, we perform any possibly scheduled draws to the root target.
            renderer.draw_pass(
                buffers
                    .draw_buffers
                    .strips
                    .ranged(&round.root_draw.strip_ranges),
                &round.root_draw.external_texture_runs,
                DrawPassTarget::Root(root_output_target),
                round
                    .root_draw
                    .has_child_layer
                    .then(|| texture_pair.layer_id(TextureParity::Odd)),
            );

            // Finally, we clear layer regions that are deallocated in this round as well as
            // all painted rectangles in the scratch buffer, so future rounds can assume a clean slate.
            for (index, (layer_clears, scratch_clears)) in round
                .layer_texture_clears
                .iter()
                .zip(round.scratch_texture_clears.iter())
                .enumerate()
            {
                let texture_parity = TextureParity::from_parity(index);
                renderer.clear_pass(
                    TextureTarget::layer_page(texture_pair.layer_id(texture_parity)),
                    layer_clears,
                );
                renderer.clear_pass(TextureTarget::scratch(texture_parity), scratch_clears);
            }
        }
    }
}
