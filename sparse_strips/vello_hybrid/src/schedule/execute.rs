// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Executes planned rendering rounds through a backend.

use super::round::{BlendOp, Rounds};
use super::{Schedule, ScheduleBuffers, ScheduleStorage};
use crate::draw::ExternalTextureRun;
use crate::filter::FilterPassPlan;
use crate::target::{
    DrawPassTarget, FilterTexturePair, LayerTextureId, LayerTexturePair, RootTarget, TextureParity,
};
use crate::util::{RangedSlice, VecExt};
use crate::{GpuStrip, blend::BlendStrip};
use vello_common::geometry::{RectU16, SizeU16};

/// A sink for rendering operations.
pub(crate) trait RendererBackend {
    /// Execute the single opaque pass against the root target with the given strips.
    ///
    /// This method is called first before any of the other ones and only once.
    fn opaque_pass(&mut self, strips: &[GpuStrip]);
    /// Execute a draw pass against the given target.
    fn draw_pass(
        &mut self,
        strips: RangedSlice<'_, GpuStrip>,
        external_texture_runs: &[ExternalTextureRun],
        target: DrawPassTarget,
        child_layer_texture: Option<LayerTextureId>,
    );
    /// Execute a clear pass with the given rectangles against the target.
    fn clear_pass(&mut self, target: LayerTextureId, rects: &[RectU16]);
    /// Execute a blend bass between a parent and child texture.
    fn blend_pass(
        &mut self,
        blends: RangedSlice<'_, BlendOp>,
        blend_strips: &[BlendStrip],
        parent_texture_parity: TextureParity,
        texture_pair: LayerTexturePair,
    );
    /// Execute a filter pass according to the filter plan between two textures.
    fn filter_pass(&mut self, plan: &FilterPassPlan, textures: FilterTexturePair);
}

pub(crate) fn execute<R: RendererBackend>(
    renderer: &mut R,
    storage: &mut ScheduleStorage,
    schedule: Schedule,
    root_output_target: RootTarget,
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
        root_output_target: RootTarget,
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
            self.texture_size,
        );
    }
}

impl Rounds {
    fn execute<R: RendererBackend>(
        &self,
        renderer: &mut R,
        root_output_target: RootTarget,
        buffers: &ScheduleBuffers,
        filter_plan: &mut FilterPassPlan,
        texture_size: SizeU16,
    ) {
        // The core loop that ties everything together!

        // We iterate over each round separately.
        for round in &self.rounds {
            let texture_pair = round.resolve_texture_binding();
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
                    texture_size,
                );
                renderer.filter_pass(
                    filter_plan,
                    FilterTexturePair::new(texture_pair, texture_parity),
                );

                // Finally, we apply all blend operations.
                renderer.blend_pass(
                    buffers.blend_ops.ranged(&pass.blend_ranges),
                    &buffers.blend_strips,
                    texture_parity,
                    texture_pair,
                );
            }

            // Once layers are done, we perform draws against the root target.
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

            // And in the end, clear layer regions that are deallocated in this round.
            for (index, layer_clears) in round.layer_texture_clears.iter().enumerate() {
                let texture_parity = TextureParity::from_parity(index);
                renderer.clear_pass(texture_pair.layer_id(texture_parity), layer_clears);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::test_support::{SceneCase, ScheduledCase};
    use alloc::vec::Vec;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Call {
        Opaque,
        Draw(DrawPassTarget, Option<LayerTextureId>),
        Filter(LayerTextureId),
        Blend(TextureParity),
        Clear(LayerTextureId),
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Stage {
        Opaque,
        EvenDraw,
        EvenFilter,
        EvenBlend,
        OddDraw,
        OddFilter,
        OddBlend,
        RootDraw,
        EvenClear,
        OddClear,
    }

    impl Call {
        fn stage(self) -> Stage {
            match self {
                Self::Opaque => Stage::Opaque,
                Self::Draw(DrawPassTarget::Layer(id), _) => match id.texture_parity {
                    TextureParity::Even => Stage::EvenDraw,
                    TextureParity::Odd => Stage::OddDraw,
                },
                Self::Draw(DrawPassTarget::Root(_), _) => Stage::RootDraw,
                Self::Filter(id) => match id.texture_parity {
                    TextureParity::Even => Stage::EvenFilter,
                    TextureParity::Odd => Stage::OddFilter,
                },
                Self::Blend(TextureParity::Even) => Stage::EvenBlend,
                Self::Blend(TextureParity::Odd) => Stage::OddBlend,
                Self::Clear(id) => match id.texture_parity {
                    TextureParity::Even => Stage::EvenClear,
                    TextureParity::Odd => Stage::OddClear,
                },
            }
        }
    }

    #[derive(Default)]
    struct Recorder {
        calls: Vec<Call>,
    }

    impl RendererBackend for Recorder {
        fn opaque_pass(&mut self, _strips: &[GpuStrip]) {
            self.calls.push(Call::Opaque);
        }

        fn draw_pass(
            &mut self,
            _strips: RangedSlice<'_, GpuStrip>,
            _external_texture_runs: &[ExternalTextureRun],
            target: DrawPassTarget,
            child_layer_texture: Option<LayerTextureId>,
        ) {
            self.calls.push(Call::Draw(target, child_layer_texture));
        }

        fn clear_pass(&mut self, target: LayerTextureId, _rects: &[RectU16]) {
            self.calls.push(Call::Clear(target));
        }

        fn blend_pass(
            &mut self,
            _blends: RangedSlice<'_, BlendOp>,
            _blend_strips: &[BlendStrip],
            parent_texture_parity: TextureParity,
            _texture_pair: LayerTexturePair,
        ) {
            self.calls.push(Call::Blend(parent_texture_parity));
        }

        fn filter_pass(&mut self, _plan: &FilterPassPlan, textures: FilterTexturePair) {
            self.calls.push(Call::Filter(textures.original()));
        }
    }

    fn layer_id(parity: TextureParity) -> LayerTextureId {
        LayerTextureId::new(parity, 0)
    }

    fn execution_case(root_target: RootTarget, depth: usize) -> ScheduledCase {
        fn chain(case: &mut SceneCase, depth: usize) {
            case.layer(|case| {
                if depth == 1 {
                    case.draw_at(4.0, 0.5);
                } else {
                    chain(case, depth - 1);
                }
            });
        }

        let mut case = SceneCase::new(16, 8);
        chain(&mut case, depth);
        case.schedule(root_target, SizeU16::new(64), 2).unwrap()
    }

    #[test]
    fn round_order() {
        let mut recorder = Recorder::default();
        execution_case(RootTarget::UserSurface, 2).execute(&mut recorder);

        assert_eq!(
            recorder
                .calls
                .into_iter()
                .map(Call::stage)
                .collect::<Vec<_>>(),
            [
                Stage::Opaque,
                Stage::EvenDraw,
                Stage::EvenFilter,
                Stage::EvenBlend,
                Stage::OddDraw,
                Stage::OddFilter,
                Stage::OddBlend,
                Stage::RootDraw,
                Stage::EvenClear,
                Stage::OddClear,
            ]
        );
    }

    #[test]
    fn two_round_order() {
        let mut recorder = Recorder::default();
        execution_case(RootTarget::UserSurface, 3).execute(&mut recorder);

        assert_eq!(
            recorder
                .calls
                .into_iter()
                .map(Call::stage)
                .collect::<Vec<_>>(),
            [
                // First round.
                Stage::Opaque,
                Stage::EvenDraw,
                Stage::EvenFilter,
                Stage::EvenBlend,
                Stage::OddDraw,
                Stage::OddFilter,
                Stage::OddBlend,
                Stage::RootDraw,
                Stage::EvenClear,
                Stage::OddClear,
                // Second round.
                Stage::EvenDraw,
                Stage::EvenFilter,
                Stage::EvenBlend,
                Stage::OddDraw,
                Stage::OddFilter,
                Stage::OddBlend,
                Stage::RootDraw,
                Stage::EvenClear,
                Stage::OddClear,
            ]
        );
    }

    #[test]
    fn child_bindings() {
        let mut recorder = Recorder::default();
        execution_case(RootTarget::UserSurface, 2).execute(&mut recorder);

        let draws = recorder
            .calls
            .into_iter()
            .filter_map(|call| match call {
                Call::Draw(target, child) => Some((target, child)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            draws,
            [
                (DrawPassTarget::Layer(layer_id(TextureParity::Even)), None,),
                (
                    DrawPassTarget::Layer(layer_id(TextureParity::Odd)),
                    Some(layer_id(TextureParity::Even)),
                ),
                (
                    DrawPassTarget::Root(RootTarget::UserSurface),
                    Some(layer_id(TextureParity::Odd)),
                ),
            ]
        );
    }

    #[test]
    fn atlas_skips_opaque() {
        let mut recorder = Recorder::default();
        execution_case(RootTarget::AtlasLayer, 2).execute(&mut recorder);

        assert!(!recorder.calls.contains(&Call::Opaque));
    }
}
