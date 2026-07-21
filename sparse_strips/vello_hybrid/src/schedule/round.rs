// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Round execution.
//!
//! Complex scenes require multiple render passes against different texture targets. Therefore,
//! rendering a single scene is split into multiple "rounds".
//!
//! A [`Round`] is the largest unit of scheduled work that can execute with one fixed pair of
//! layer texture pages: one page from the even texture group and one from the odd texture group.
//! It contains the following work:
//!
//! - One [`LayerTexturePass`] for the even page.
//! - One [`LayerTexturePass`] for the odd page.
//! - A [`Draw`] targeting the root output.
//! - Rectangles to clear in each layer page after the rendering work completes.
//!
//! ## Texture bindings
//!
//! All layer work in a round uses the same [`RoundBindings`]. Before placing an operation in a
//! round, the scheduler determines which layer texture pages the operation needs to render to or
//! sample from. The operation can share the round only if those pages are compatible with the
//! round's existing bindings. Operations that require only the even or only the odd page can still
//! batch together. If two operations require different pages of the same parity, their bindings
//! conflict and the later operation is placed in a subsequent compatible round.

use super::ScheduleBuffers;
use crate::draw::{Draw, ExternalTextureRun};
use crate::filter::GpuFilterData;
use crate::target::{
    BlendPassBindings, DrawPassBindings, DrawPassTarget, FilterPassBindings, LayerTextureId,
    LayerTextureRegion, RootTarget, RoundBindings, TextureParity, TextureRegion,
};
use crate::util::{RangedSlice, Ranges, VecExt};
use crate::{GpuStrip, blend::BlendStrip};
use alloc::vec::Vec;
use core::ops::Range;
use vello_common::geometry::RectU16;
use vello_common::peniko::BlendMode;

/// Completed rounds and the layer texture pages required to execute them.
#[derive(Debug, Default)]
pub(super) struct Rounds {
    /// Rounds in execution order.
    pub(super) rounds: Vec<Round>,
    /// Required page count for the even and odd texture groups.
    layer_page_counts: [usize; 2],
}

impl Rounds {
    pub(super) fn iter(&self) -> core::slice::Iter<'_, Round> {
        self.rounds.iter()
    }

    pub(super) const fn layer_page_counts(&self) -> [usize; 2] {
        self.layer_page_counts
    }

    pub(super) fn round_mut(&mut self, index: usize) -> &mut Round {
        &mut self.rounds[index]
    }

    pub(super) fn push_layer_clear(
        &mut self,
        round_idx: usize,
        texture_parity: TextureParity,
        rect: RectU16,
    ) {
        let parity = texture_parity.get_parity();

        self.rounds[round_idx].layer_texture_clears[parity].push(rect);
    }

    pub(super) fn require_layer_texture(&mut self, texture: LayerTextureId) {
        let required = usize::from(texture.page_index) + 1;
        let page_count = &mut self.layer_page_counts[texture.texture_parity.get_parity()];
        *page_count = (*page_count).max(required);
    }

    pub(super) fn ensure_exists(&mut self, round_idx: usize) {
        while self.rounds.len() <= round_idx {
            self.rounds.push(Round::default());
        }
    }

    pub(super) fn resolve_binding_point(
        &mut self,
        mut point: SchedulePoint,
        requirement: RoundBindings,
    ) -> SchedulePoint {
        // This should be more than enough for any sane scene. If it turns out not, we
        // can always just remove the guard since, _in theory_, it should never be possible
        // to trigger an endless loop.
        const MAX_ROUNDS: usize = 40_000;

        for texture in requirement.required_textures().into_iter().flatten() {
            self.require_layer_texture(texture);
        }

        loop {
            // This shouldn't ever happen, unless there is some kind of logic bug. But better
            // to panic at some point than loop on forever.
            // TODO: Turn this into an error once we have refactored error handling.
            if point.round > MAX_ROUNDS {
                panic!("possible deadlock in scheduler detected");
            }

            self.ensure_exists(point.round);

            let round = &mut self.rounds[point.round];

            // If the given round has a compatible texture binding, we can fold
            // into it.
            if let Some(binding) = round.texture_binding.merge(requirement) {
                round.texture_binding = binding;

                return point;
            }

            // Otherwise, keep looking.
            point.round += 1;
        }
    }
}

/// Operations and texture binding requirements for one rendering round.
#[derive(Debug, Default)]
pub(super) struct Round {
    /// Page required from each layer texture parity.
    pub(super) texture_binding: RoundBindings,
    /// Draw, filter, and blend work for the even and odd layer textures.
    layer_texture_passes: [LayerTexturePass; 2],
    /// Draw targeting the root output after both layer passes.
    root_draw: Draw,
    /// Regions cleared after all rendering work in this round.
    layer_texture_clears: [Vec<RectU16>; 2],
}

impl Round {
    pub(super) fn layer_passes<'a>(
        &'a self,
        buffers: &'a ScheduleBuffers,
    ) -> impl Iterator<Item = LayerPass<'a>> + 'a {
        self.layer_texture_passes
            .iter()
            .enumerate()
            .filter_map(move |(index, pass)| {
                let texture_parity = TextureParity::from_parity(index);
                let target = self.texture_binding.layer_id(texture_parity);
                let opposite = self.texture_binding.layer_id(texture_parity.opposite());

                let draw = (pass.draw.strip_ranges.len() != 0).then(|| DrawPass {
                    strips: buffers.draw_buffers.strips.ranged(&pass.draw.strip_ranges),
                    external_texture_runs: &pass.draw.external_texture_runs,
                    bindings: DrawPassBindings::new(
                        DrawPassTarget::Layer(target.unwrap()),
                        pass.draw.has_child_layer.then(|| opposite.unwrap()),
                    ),
                });

                let filter = (pass.filter_ranges.len() != 0).then(|| FilterPass {
                    filters: buffers.filter_ops.ranged(&pass.filter_ranges),
                    bindings: FilterPassBindings::new(target.unwrap(), opposite.unwrap()),
                });

                let blend = (pass.blend_ranges.len() != 0).then(|| BlendPass {
                    blends: buffers.blend_ops.ranged(&pass.blend_ranges),
                    blend_strips: &buffers.blend_strips,
                    bindings: BlendPassBindings::new(target.unwrap(), opposite.unwrap()),
                });

                (draw.is_some() || filter.is_some() || blend.is_some()).then_some(LayerPass {
                    texture_parity,
                    draw,
                    filter,
                    blend,
                })
            })
    }

    pub(super) fn root_draw_pass<'a>(
        &'a self,
        buffers: &'a ScheduleBuffers,
        target: RootTarget,
    ) -> Option<DrawPass<'a>> {
        if self.root_draw.strip_ranges.len() == 0 {
            return None;
        }

        let child = self
            .root_draw
            .has_child_layer
            .then(|| self.texture_binding.layer_id(TextureParity::Odd).unwrap());

        Some(DrawPass {
            strips: buffers
                .draw_buffers
                .strips
                .ranged(&self.root_draw.strip_ranges),
            external_texture_runs: &self.root_draw.external_texture_runs,
            bindings: DrawPassBindings::new(DrawPassTarget::Root(target), child),
        })
    }

    pub(super) fn clear_passes(&self) -> impl Iterator<Item = ClearPass<'_>> {
        self.layer_texture_clears
            .iter()
            .enumerate()
            .filter_map(|(index, rects)| {
                if rects.is_empty() {
                    return None;
                }

                let texture_parity = TextureParity::from_parity(index);
                Some(ClearPass {
                    target: self.texture_binding.layer_id(texture_parity).unwrap(),
                    rects,
                })
            })
    }

    pub(super) fn root_draw_mut(&mut self) -> &mut Draw {
        &mut self.root_draw
    }

    pub(super) fn layer_draw_mut(&mut self, texture_parity: TextureParity) -> &mut Draw {
        &mut self.layer_texture_passes[texture_parity.get_parity()].draw
    }

    pub(super) fn push_blend_op(
        &mut self,
        parent_texture_parity: TextureParity,
        buffers: &mut ScheduleBuffers,
        blend: BlendOp,
    ) {
        buffers.blend_ops.push_ranged(
            &mut self.layer_texture_passes[parent_texture_parity.get_parity()].blend_ranges,
            blend,
        );
    }

    pub(super) fn push_filter_op(
        &mut self,
        texture_parity: TextureParity,
        buffers: &mut ScheduleBuffers,
        filter: FilterOp,
    ) {
        buffers.filter_ops.push_ranged(
            &mut self.layer_texture_passes[texture_parity.get_parity()].filter_ranges,
            filter,
        );
    }
}

// Note that the field order of these enums matters since we implement
// `PartialOrd` and use this to represent the order in which the stages
// happen.

/// A stage in the execution of a rendering round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum RoundStage {
    /// Start of a round, before any rendering stage.
    Start,
    /// Draw, filter, or blend stage targeting the even layer texture.
    Even(LayerStage),
    /// Draw, filter, or blend stage targeting the odd layer texture.
    Odd(LayerStage),
    /// Draw stage targeting the root output.
    RootDraw,
}

/// Stages executed for one parity of the layer texture pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum LayerStage {
    /// Render recorded draws.
    Draw,
    /// Apply filter passes.
    Filter,
    /// Apply non-default blend operations.
    Blend,
}

impl RoundStage {
    pub(super) const fn draw(texture_parity: TextureParity) -> Self {
        match texture_parity {
            TextureParity::Even => Self::Even(LayerStage::Draw),
            TextureParity::Odd => Self::Odd(LayerStage::Draw),
        }
    }

    pub(super) const fn filter(texture_parity: TextureParity) -> Self {
        match texture_parity {
            TextureParity::Even => Self::Even(LayerStage::Filter),
            TextureParity::Odd => Self::Odd(LayerStage::Filter),
        }
    }

    pub(super) const fn blend(texture_parity: TextureParity) -> Self {
        match texture_parity {
            TextureParity::Even => Self::Even(LayerStage::Blend),
            TextureParity::Odd => Self::Odd(LayerStage::Blend),
        }
    }
}

// As for `RoundStage`, the order of fields here is important!
/// A precise point in the execution timeline of the schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct SchedulePoint {
    /// Index of the round containing this point.
    pub(super) round: usize,
    /// Stage within the round.
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
    pub(super) fn after(self, stage: RoundStage) -> Self {
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

    /// Return the first occurrence of `stage` at or after this point.
    pub(super) fn after_or_at(self, stage: RoundStage) -> Self {
        if stage >= self.stage {
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

/// Work targeting one parity of the layer texture pair in a round.
#[derive(Debug, Default)]
pub(super) struct LayerTexturePass {
    /// Strip draw executed at the start of the layer pass.
    draw: Draw,
    /// Ranges of filter operations executed after the draw.
    filter_ranges: Ranges,
    /// Ranges of blend operations executed after the filters.
    blend_ranges: Ranges,
}

/// Draw work and resolved texture bindings for one pass.
#[derive(Debug)]
pub(super) struct DrawPass<'a> {
    pub(super) strips: RangedSlice<'a, GpuStrip>,
    pub(super) external_texture_runs: &'a [ExternalTextureRun],
    pub(super) bindings: DrawPassBindings,
}

/// Filter work and resolved texture bindings for one pass.
#[derive(Debug)]
pub(super) struct FilterPass<'a> {
    pub(super) filters: RangedSlice<'a, FilterOp>,
    pub(super) bindings: FilterPassBindings,
}

/// Blend work and resolved texture bindings for one pass.
#[derive(Debug)]
pub(super) struct BlendPass<'a> {
    pub(super) blends: RangedSlice<'a, BlendOp>,
    pub(super) blend_strips: &'a [BlendStrip],
    pub(super) bindings: BlendPassBindings,
}

/// Clear work and its resolved texture target.
#[derive(Debug)]
pub(super) struct ClearPass<'a> {
    pub(super) target: LayerTextureId,
    pub(super) rects: &'a [RectU16],
}

/// Executable work for one non-empty layer-texture pass.
#[derive(Debug)]
pub(super) struct LayerPass<'a> {
    pub(super) draw: Option<DrawPass<'a>>,
    pub(super) filter: Option<FilterPass<'a>>,
    pub(super) blend: Option<BlendPass<'a>>,
    #[allow(dead_code, reason = "only used for a test.")]
    pub(super) texture_parity: TextureParity,
}

/// Original and temporary regions used to ping-pong a filter sequence.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FilterTextureRegions {
    /// Region containing the input and final filtered result.
    pub(crate) original: TextureRegion,
    /// Opposite-parity region used for intermediate passes.
    pub(crate) temporary: TextureRegion,
}

impl FilterTextureRegions {
    pub(crate) fn new(original: TextureRegion, temporary: TextureRegion) -> Self {
        Self {
            original,
            temporary,
        }
    }

    pub(crate) fn round_bindings(self) -> RoundBindings {
        RoundBindings::new(self.original.target)
            .merge(RoundBindings::new(self.temporary.target))
            .unwrap()
    }
}

/// A scheduled filter and the texture regions on which it operates.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FilterOp {
    /// Original and temporary regions used by the filter passes.
    pub(crate) textures: FilterTextureRegions,
    /// Texel offset of this filter's parameters in the filter data texture.
    pub(crate) filter_data_offset: u32,
    /// Prepared filter parameters used to select and size passes.
    pub(crate) gpu_filter: GpuFilterData,
}

/// A scheduled non-default blend between a parent and child layer.
#[derive(Debug, Clone)]
pub(crate) struct BlendOp {
    /// Parent layer serving as both backdrop and blend destination.
    pub(crate) parent_region: LayerTextureRegion,
    /// Child layer serving as the blend source.
    pub(crate) child_region: LayerTextureRegion,
    /// Scene-space bounds affected by the blend.
    pub(crate) blend_bbox: RectU16,
    /// The blend mode that should be applied.
    pub(crate) blend_mode: BlendMode,
    /// Opacity applied to the child before sampling.
    pub(crate) opacity: f32,
    /// Range of strips used for clipping the blend layer. If `None`, the blend spans its bbox.
    pub(crate) clip_strips: Option<Range<u32>>,
}

#[cfg(test)]
mod tests {
    use super::{
        BlendOp, FilterOp, FilterTextureRegions, LayerStage, Round, RoundStage, Rounds,
        SchedulePoint,
    };
    use crate::filter::GpuFilterData;
    use crate::schedule::ScheduleBuffers;
    use crate::target::{
        LayerTextureId, LayerTextureRegion, RoundBindings, TextureParity, TextureRegion,
    };
    use crate::util::VecExt;
    use bytemuck::Zeroable;
    use vello_common::geometry::RectU16;
    use vello_common::peniko::BlendMode;

    fn layer_id(texture_parity: TextureParity, page_index: u16) -> LayerTextureId {
        LayerTextureId::new(texture_parity, page_index)
    }

    fn region(texture_parity: TextureParity, page_index: u16) -> TextureRegion {
        TextureRegion {
            target: layer_id(texture_parity, page_index),
            rect: RectU16::new(0, 0, 16, 16),
        }
    }

    fn layer_region(texture_parity: TextureParity, page_index: u16) -> LayerTextureRegion {
        LayerTextureRegion {
            texture: region(texture_parity, page_index),
            layer_bbox: RectU16::new(0, 0, 16, 16),
        }
    }

    fn filter_op(filter_data_offset: u32) -> FilterOp {
        FilterOp {
            textures: FilterTextureRegions::new(
                region(TextureParity::Even, 0),
                region(TextureParity::Odd, 0),
            ),
            filter_data_offset,
            gpu_filter: GpuFilterData::zeroed(),
        }
    }

    #[test]
    fn schedule_point_stage_occurrences() {
        let draw = SchedulePoint {
            round: 4,
            stage: RoundStage::Even(LayerStage::Draw),
        };

        assert_eq!(
            draw.after(RoundStage::Even(LayerStage::Filter)),
            SchedulePoint {
                round: 4,
                stage: RoundStage::Even(LayerStage::Filter),
            }
        );
        assert_eq!(
            draw.after(RoundStage::Odd(LayerStage::Draw)),
            SchedulePoint {
                round: 4,
                stage: RoundStage::Odd(LayerStage::Draw),
            }
        );
        assert_eq!(
            draw.after(RoundStage::Even(LayerStage::Draw)),
            SchedulePoint {
                round: 5,
                stage: RoundStage::Even(LayerStage::Draw),
            }
        );
        assert_eq!(draw.after_or_at(RoundStage::Even(LayerStage::Draw)), draw);
        assert_eq!(
            draw.after_or_at(RoundStage::Start),
            SchedulePoint {
                round: 5,
                stage: RoundStage::Start,
            }
        );
    }

    #[test]
    fn unused_bindings_do_not_resolve_passes() {
        let round = Round {
            texture_binding: RoundBindings::new(layer_id(TextureParity::Even, 2))
                .merge(RoundBindings::new(layer_id(TextureParity::Odd, 5)))
                .unwrap(),
            ..Round::default()
        };

        assert_eq!(round.layer_passes(&ScheduleBuffers::default()).count(), 0);
        assert_eq!(round.clear_passes().count(), 0);
    }

    #[test]
    fn pass_iterators_skip_empty_parities() {
        let even = layer_id(TextureParity::Even, 2);
        let odd = layer_id(TextureParity::Odd, 5);
        let mut round = Round {
            texture_binding: RoundBindings::new(even)
                .merge(RoundBindings::new(odd))
                .unwrap(),
            ..Round::default()
        };

        let mut buffers = ScheduleBuffers::default();
        round.push_filter_op(TextureParity::Even, &mut buffers, filter_op(10));
        round.layer_texture_clears[TextureParity::Odd.get_parity()].push(RectU16::new(0, 0, 8, 8));

        let layer_passes = round.layer_passes(&buffers).collect::<alloc::vec::Vec<_>>();
        assert_eq!(layer_passes.len(), 1);
        assert!(layer_passes[0].draw.is_none());
        assert_eq!(
            layer_passes[0].filter.as_ref().unwrap().bindings.target(),
            even
        );
        assert!(layer_passes[0].blend.is_none());

        let clear_passes = round.clear_passes().collect::<alloc::vec::Vec<_>>();
        assert_eq!(clear_passes.len(), 1);
        assert_eq!(clear_passes[0].target, odd);
    }

    #[test]
    fn binding_conflicts_even() {
        let mut rounds = Rounds::default();
        let requested = SchedulePoint {
            round: 0,
            stage: RoundStage::Even(LayerStage::Draw),
        };

        let even_page_1 = RoundBindings::new(layer_id(TextureParity::Even, 1));
        let odd_page_2 = RoundBindings::new(layer_id(TextureParity::Odd, 2));
        let even_page_3 = RoundBindings::new(layer_id(TextureParity::Even, 3));

        // First round gets assigned even page 1.
        assert_eq!(
            rounds.resolve_binding_point(requested, even_page_1),
            requested
        );

        // First round gets assigned odd page 2.
        assert_eq!(
            rounds.resolve_binding_point(requested, odd_page_2),
            requested
        );

        // First round already has even page binding, so we much advance
        // to next round.
        assert_eq!(
            rounds.resolve_binding_point(requested, even_page_3),
            SchedulePoint {
                round: 1,
                ..requested
            }
        );

        assert_eq!(
            rounds.rounds[0].texture_binding.page_indices(),
            [Some(1), Some(2)]
        );
        assert_eq!(
            rounds.rounds[1].texture_binding.page_indices(),
            [Some(3), None]
        );
        assert_eq!(rounds.layer_page_counts, [4, 3]);
    }

    #[test]
    fn binding_conflicts_odd() {
        let mut rounds = Rounds::default();
        let requested = SchedulePoint {
            round: 0,
            stage: RoundStage::Odd(LayerStage::Filter),
        };

        let even_page_1 = RoundBindings::new(layer_id(TextureParity::Even, 1));
        let odd_page_2 = RoundBindings::new(layer_id(TextureParity::Odd, 2));
        let odd_page_4 = RoundBindings::new(layer_id(TextureParity::Odd, 4));

        assert_eq!(
            rounds.resolve_binding_point(requested, even_page_1),
            requested
        );
        assert_eq!(
            rounds.resolve_binding_point(requested, odd_page_2),
            requested
        );
        assert_eq!(
            rounds.resolve_binding_point(requested, odd_page_4),
            SchedulePoint {
                round: 1,
                ..requested
            }
        );

        assert_eq!(
            rounds.rounds[0].texture_binding.page_indices(),
            [Some(1), Some(2)]
        );
        assert_eq!(
            rounds.rounds[1].texture_binding.page_indices(),
            [None, Some(4)]
        );
        assert_eq!(rounds.layer_page_counts, [2, 5]);
    }

    #[test]
    fn binding_conflicts_skip_rounds() {
        let mut rounds = Rounds::default();
        let point = |round| SchedulePoint {
            round,
            stage: RoundStage::RootDraw,
        };

        rounds.resolve_binding_point(
            point(0),
            RoundBindings::new(layer_id(TextureParity::Even, 0)),
        );

        rounds.resolve_binding_point(
            point(1),
            RoundBindings::new(layer_id(TextureParity::Even, 1)),
        );

        assert_eq!(
            rounds.resolve_binding_point(
                point(0),
                RoundBindings::new(layer_id(TextureParity::Even, 2)),
            ),
            point(2)
        );
        assert_eq!(rounds.rounds.len(), 3);
        assert_eq!(rounds.layer_page_counts, [3, 0]);
    }

    #[test]
    fn operation_ranges() {
        let mut rounds = Rounds::default();
        let mut buffers = ScheduleBuffers::default();
        rounds.ensure_exists(0);

        let round = &mut rounds.rounds[0];
        round.push_filter_op(TextureParity::Even, &mut buffers, filter_op(10));
        round.push_filter_op(TextureParity::Odd, &mut buffers, filter_op(20));
        round.push_filter_op(TextureParity::Even, &mut buffers, filter_op(30));
        round.push_blend_op(
            TextureParity::Odd,
            &mut buffers,
            BlendOp {
                parent_region: layer_region(TextureParity::Odd, 0),
                child_region: layer_region(TextureParity::Even, 0),
                blend_bbox: RectU16::new(0, 0, 16, 16),
                blend_mode: BlendMode::default(),
                opacity: 1.0,
                clip_strips: None,
            },
        );

        let even_clear_1 = RectU16::new(0, 0, 8, 8);
        let odd_clear = RectU16::new(8, 0, 16, 8);
        let even_clear_2 = RectU16::new(0, 8, 8, 16);
        rounds.push_layer_clear(0, TextureParity::Even, even_clear_1);
        rounds.push_layer_clear(0, TextureParity::Odd, odd_clear);
        rounds.push_layer_clear(0, TextureParity::Even, even_clear_2);

        let round = &rounds.rounds[0];
        let even_pass = &round.layer_texture_passes[TextureParity::Even.get_parity()];
        let odd_pass = &round.layer_texture_passes[TextureParity::Odd.get_parity()];
        let even_offsets: alloc::vec::Vec<_> = buffers
            .filter_ops
            .ranged(&even_pass.filter_ranges)
            .iter()
            .map(|op| op.filter_data_offset)
            .collect();
        let odd_offsets: alloc::vec::Vec<_> = buffers
            .filter_ops
            .ranged(&odd_pass.filter_ranges)
            .iter()
            .map(|op| op.filter_data_offset)
            .collect();

        assert_eq!(even_offsets, [10, 30]);
        assert_eq!(odd_offsets, [20]);
        assert_eq!(even_pass.blend_ranges.len(), 0);
        assert_eq!(odd_pass.blend_ranges.len(), 1);
        assert_eq!(
            round.layer_texture_clears[TextureParity::Even.get_parity()],
            [even_clear_1, even_clear_2]
        );
        assert_eq!(
            round.layer_texture_clears[TextureParity::Odd.get_parity()],
            [odd_clear]
        );
    }
}
