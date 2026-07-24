// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Executes planned rendering rounds through a backend.

use super::round::{BlendOp, Rounds};
use super::{Schedule, ScheduleBuffers, ScheduleStorage};
use crate::draw::ExternalTextureRun;
use crate::filter::FilterPassPlan;
use crate::target::{
    BlendPassBindings, DrawPassBindings, DrawPassTarget, FilterPassBindings, LayerTextureId,
    RootTarget,
};
use crate::util::RangedSlice;
use crate::{GpuStrip, blend::BlendStrip};
use vello_common::geometry::{RectU16, SizeU16};

/// A backend for executing GPU render passes.
pub(crate) trait Backend {
    /// Execute the opaque pass against the root target with the given strips. If this method is
    /// ever called, it's called before any of the other ones and only once.
    ///
    /// The strips are guaranteed to be non-empty.
    fn opaque_pass(&mut self, strips: &[GpuStrip]);

    /// Execute a draw pass against the given target.
    ///
    /// The strips are guaranteed to be non-empty.
    fn draw_pass(
        &mut self,
        strips: RangedSlice<'_, GpuStrip>,
        external_texture_runs: &[ExternalTextureRun],
        bindings: DrawPassBindings,
    );

    /// Execute a clear pass with the given rectangles against the target.
    ///
    /// The clear rectangles are guaranteed to be non-empty.
    fn clear_pass(&mut self, target: LayerTextureId, rects: &[RectU16]);

    /// Execute a blend pass between a parent and child texture.
    ///
    /// The blends are guaranteed to be non-empty.
    fn blend_pass(
        &mut self,
        blends: RangedSlice<'_, BlendOp>,
        blend_strips: &[BlendStrip],
        bindings: BlendPassBindings,
    );

    /// Execute a filter pass according to the filter plan between two textures.
    ///
    /// The filter pass plan is guaranteed to be non-empty.
    fn filter_pass(&mut self, plan: &FilterPassPlan, bindings: FilterPassBindings);
}

pub(crate) fn execute<R: Backend>(
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
    fn execute<R: Backend>(
        &self,
        renderer: &mut R,
        root_output_target: RootTarget,
        buffers: &ScheduleBuffers,
        filter_plan: &mut FilterPassPlan,
    ) {
        if DrawPassTarget::Root(root_output_target).enable_opaque()
            && !buffers.draw_buffers.opaque_strips.is_empty()
        {
            renderer.opaque_pass(&buffers.draw_buffers.opaque_strips);
        }

        self.rounds.execute(
            renderer,
            root_output_target,
            buffers,
            filter_plan,
            self.intermediate_textures.size,
        );
    }
}

impl Rounds {
    fn execute<R: Backend>(
        &self,
        backend: &mut R,
        root_output_target: RootTarget,
        buffers: &ScheduleBuffers,
        filter_plan: &mut FilterPassPlan,
        texture_size: SizeU16,
    ) {
        // This is the core loop that ties everything together.

        // TODO: Currently, we upload data for layers when executing each round. It might be
        // worth exploring whether we can upload the data once in the beginning, and then
        // reference it via slices.

        // We iterate over each round separately.
        for round in self.iter() {
            // Draw the even layer texture before the odd one. A child layer always lives in the opposite
            // group from its parent, and only odd-texture draws sample the even texture (never the reverse
            // within a round), so this order guarantees those pixels are ready when we read them.

            for layer_passes in round.layer_passes(buffers) {
                // For each layer texture target, first perform the draws of all layers that are
                // allocated in this texture.
                if let Some(pass) = layer_passes.draw {
                    backend.draw_pass(pass.strips, pass.external_texture_runs, pass.bindings);
                }

                // Next, we apply all filters for layers in this texture.
                if let Some(pass) = layer_passes.filter {
                    filter_plan.init(pass.filters.iter().copied(), texture_size);

                    backend.filter_pass(filter_plan, pass.bindings);
                }

                // Finally, we apply all blend operations of layers in the current texture
                // that form the backdrop of some child texture.
                if let Some(pass) = layer_passes.blend {
                    backend.blend_pass(pass.blends, pass.blend_strips, pass.bindings);
                }
            }

            // Once layers are done, we perform draws against the root target.
            if let Some(pass) = round.root_draw_pass(buffers, root_output_target) {
                backend.draw_pass(pass.strips, pass.external_texture_runs, pass.bindings);
            }

            // And in the end, clear all layer regions that are deallocated in this round.
            for props in round.clear_passes() {
                backend.clear_pass(props.target, props.rects);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::test_support::{SceneCase, ScheduledCase};
    use crate::target::{LayerTextureId, TextureParity};
    use alloc::vec::Vec;
    use vello_common::filter_effects::{Filter, FilterPrimitive};
    use vello_common::peniko::{BlendMode, Compose, Mix};

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

    impl Backend for Recorder {
        fn opaque_pass(&mut self, _strips: &[GpuStrip]) {
            self.calls.push(Call::Opaque);
        }

        fn draw_pass(
            &mut self,
            _strips: RangedSlice<'_, GpuStrip>,
            _external_texture_runs: &[ExternalTextureRun],
            bindings: DrawPassBindings,
        ) {
            self.calls.push(Call::Draw(bindings.target, bindings.child));
        }

        fn clear_pass(&mut self, target: LayerTextureId, _rects: &[RectU16]) {
            self.calls.push(Call::Clear(target));
        }

        fn blend_pass(
            &mut self,
            _blends: RangedSlice<'_, BlendOp>,
            _blend_strips: &[BlendStrip],
            bindings: BlendPassBindings,
        ) {
            self.calls
                .push(Call::Blend(bindings.target().texture_parity));
        }

        fn filter_pass(&mut self, _plan: &FilterPassPlan, bindings: FilterPassBindings) {
            self.calls.push(Call::Filter(bindings.target()));
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
        case.draw_at(0.0, 1.0);
        chain(&mut case, depth);
        case.schedule(root_target, SizeU16::new(64), 2).unwrap()
    }

    fn round_order_case() -> ScheduledCase {
        let mut case = SceneCase::new(32, 8);
        case.draw_at(0.0, 1.0);

        case.layer(|case| {
            case.draw_at(4.0, 0.5);
            case.layer_with(
                None,
                Some(BlendMode::new(Mix::Multiply, Compose::SrcOver)),
                Some(Filter::from_primitive(FilterPrimitive::Offset {
                    dx: 0.0,
                    dy: 0.0,
                })),
                |case| case.draw_at(8.0, 0.5),
            );
        });

        case.layer(|case| {
            case.layer(|case| {
                case.draw_at(16.0, 0.5);
                case.layer_with(
                    None,
                    Some(BlendMode::new(Mix::Multiply, Compose::SrcOver)),
                    Some(Filter::from_primitive(FilterPrimitive::Offset {
                        dx: 0.0,
                        dy: 0.0,
                    })),
                    |case| case.draw_at(20.0, 0.5),
                );
            });
        });

        case.schedule(RootTarget::UserSurface, SizeU16::new(128), 6)
            .unwrap()
    }

    #[test]
    fn round_order() {
        let mut recorder = Recorder::default();
        round_order_case().execute(&mut recorder);

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
                Stage::OddDraw,
                Stage::OddFilter,
                Stage::OddBlend,
                Stage::RootDraw,
                Stage::EvenClear,
                Stage::OddClear,
                // Second round.
                Stage::EvenBlend,
                Stage::OddDraw,
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

    #[test]
    fn empty_passes_are_skipped() {
        let mut recorder = Recorder::default();
        SceneCase::new(16, 8).schedule_root().execute(&mut recorder);

        assert!(recorder.calls.is_empty());
    }
}
