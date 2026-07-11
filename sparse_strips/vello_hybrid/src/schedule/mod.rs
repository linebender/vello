// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Builds and executes dependency-ordered rendering rounds for `vello_hybrid`.

mod allocate;
mod cursor;
pub(crate) mod execute;
pub(crate) mod round;

use self::allocate::{Atlases, LayerAllocationRequest, LayerAllocations};
use self::cursor::Cursor;
pub(crate) use self::execute::{RendererBackend, execute};
use self::round::{BlendOp, FilterOp, Round, Rounds};
use crate::blend::BLEND_SCRATCH_INDEX;
use crate::draw::{Draw, DrawBuffers, DrawBuilder, DrawState, LayerSample};
use crate::filter::{FilterContext, FilterPassPlan, PreparedGpuFilter};
use crate::paint::PaintResolver;
use crate::scene::RecordedDraw;
use crate::target::{
    DrawTarget, IntermediateTextureSizes, LayerTextureRegion, RootRenderTarget, TextureIndex,
    TextureRegion,
};
use crate::{RenderError, Scene};
use alloc::vec::Vec;
use vello_common::filter::FilterLayerPlacement;
use vello_common::geometry::RectU16;
use vello_common::peniko::BlendMode;
use vello_common::record::{
    CmdNode, CommandRecorder, LayerProps, RecordedLayer, RecordedLayerKind,
};
use vello_common::strip_generator::StripStorage;
use vello_common::util::RectExt;

const REGULAR_LAYER_KIND: RecordedLayerKind = RecordedLayerKind::Regular;

#[derive(Debug)]
pub(crate) struct Schedule {
    rounds: Rounds,
    texture_sizes: IntermediateTextureSizes,
}

impl Schedule {
    pub(crate) fn try_new(
        storage: &mut ScheduleStorage,
        scene: &Scene,
        root_output_target: RootRenderTarget,
        paint_resolver: PaintResolver<'_>,
        texture_sizes: IntermediateTextureSizes,
    ) -> Result<Self, RenderError> {
        storage.clear();

        let strip_storage = scene.strip_storage.borrow();
        let scene_bbox = RectU16::new(0, 0, scene.width, scene.height).snap_to_tile_coordinates();

        let scheduler = Scheduler::new(
            &scene.recorder,
            scene_bbox,
            &strip_storage,
            root_output_target,
            paint_resolver,
            texture_sizes,
            storage,
        );

        scheduler.build()
    }
}

// TODO: Explain how the scheduling algorithm works.

/// Plans concrete, executable rounds from a recorded scene.
#[derive(Debug)]
struct Scheduler<'a, 'p> {
    recorder: &'a CommandRecorder<RecordedDraw>,
    scene_bbox: RectU16,
    strip_storage: &'a StripStorage,
    root_render_target: RootRenderTarget,
    paint_resolver: PaintResolver<'a>,
    cursor: Cursor<Atlases>,
    unreleased_layer_count: usize,
    texture_sizes: IntermediateTextureSizes,
    storage: &'p mut ScheduleStorage,
}

impl<'a, 'p> Scheduler<'a, 'p> {
    fn new(
        recorder: &'a CommandRecorder<RecordedDraw>,
        scene_bbox: RectU16,
        strip_storage: &'a StripStorage,
        root_render_target: RootRenderTarget,
        paint_resolver: PaintResolver<'a>,
        texture_sizes: IntermediateTextureSizes,
        storage: &'p mut ScheduleStorage,
    ) -> Self {
        Self {
            recorder,
            scene_bbox,
            strip_storage,
            root_render_target,
            paint_resolver,
            cursor: Cursor::new(Atlases::new(texture_sizes)),
            unreleased_layer_count: 0,
            texture_sizes,
            storage,
        }
    }

    fn build(mut self) -> Result<Schedule, RenderError> {
        let mut rounds = Rounds::default();
        self.schedule_root(&mut rounds)?;

        assert_eq!(
            self.unreleased_layer_count, 0,
            "all layers should have been released"
        );

        // Since the strips should be rendered front-to-back.
        self.storage.buffers.draw_buffers.opaque_strips.reverse();

        Ok(Schedule {
            rounds,
            texture_sizes: self.texture_sizes,
        })
    }

    fn schedule_root(&mut self, rounds: &mut Rounds) -> Result<(), RenderError> {
        let target = self.root_render_target;

        if self.recorder.root_is_blend_target {
            // If the layer is a target of a non-default blending operation, we need to be able to
            // sample from it. However, this is not possible if we render directly into the
            // user-provided view. Therefore, we need to simulate a layer push, do all the rendering
            // there and then blit back into the main frame buffer.

            let opened_layer = self.open_root_layer();
            let layer = self.schedule_layer(opened_layer, rounds)?;
            let mut state = TargetScheduleState::new(target, layer.ready_round, self.scene_bbox);

            rounds.build_draw(
                &mut state,
                &mut self.storage.buffers.draw_buffers,
                |builder| {
                    builder.push_layer_fill(layer.sample, 1.0, None, self.strip_storage);
                },
            );

            let ready_round = layer.ready_round;
            self.release_layer(layer, ready_round, rounds);
        } else {
            let mut state =
                TargetScheduleState::new(target, self.cursor.current_round(), self.scene_bbox);

            for cmd in &self.recorder.root_cmds {
                // Remember: Each command node consists of a sequence of draws + an option layer invocation.

                // First, we schedule the layer node. This might trigger advances to our current base round.
                let child = self.prepare_node(cmd, state.draw_state.target_bbox, rounds)?;

                // Then, we just submit all draws to the root output target for whatever round we are
                // currently in.
                self.push_draws(&cmd.draws, &mut state, rounds);

                // Finally, we also schedule the layer sampling operation.
                if let Some(child) = child {
                    self.compose_simple_layer(child.props, child.layer, &mut state, rounds);
                }
            }
        };

        Ok(())
    }

    fn prepare_node(
        &mut self,
        cmd: &CmdNode,
        parent_bounds: RectU16,
        rounds: &mut Rounds,
    ) -> Result<Option<PreparedChild<'a>>, RenderError> {
        let Some(layer_id) = cmd.layer else {
            return Ok(None);
        };

        let layer = &self.recorder.layers[layer_id as usize];

        let bbox = if layer.bbox.is_empty() {
            if layer.props.blend_mode.is_destructive() {
                // Unlike in the non-destructive case, empty *destructive* layers are
                // not a no-op. Instead, they clear the whole parent layer. Therefore, we
                // need to set an explicit bounding box instead of keeping an empty one,
                // as a workaround since we cannot allocate a 0x0 area in the atlas.
                // TODO: Properly handle clipped blend layers.
                parent_bounds
            } else {
                // TODO: Prune empty layers at the recording layer, so we don't need
                // this here.
                return Ok(None);
            }
        } else {
            layer.bbox
        };

        let opened_layer = self.open_layer(layer, bbox);
        let scheduled = self.schedule_layer(opened_layer, rounds)?;
        Ok(Some(PreparedChild {
            props: &layer.props,
            layer: scheduled,
        }))
    }

    fn schedule_node(
        &mut self,
        cmd: &CmdNode,
        child: Option<PreparedChild<'a>>,
        state: &mut TargetScheduleState<LayerTextureRegion>,
        rounds: &mut Rounds,
    ) {
        self.push_draws(&cmd.draws, state, rounds);

        if let Some(child) = child {
            self.compose_layer(child.props, child.layer, state, rounds);
        }
    }

    fn open_layer(&self, layer: &'a RecordedLayer, bbox: RectU16) -> OpenLayer<'a> {
        let sample = match &layer.kind {
            RecordedLayerKind::Regular => LayerSamplePlacement::regular(bbox),
            RecordedLayerKind::Filter { placement, .. } => LayerSamplePlacement::filter(*placement),
        };

        OpenLayer {
            cmds: &layer.cmds,
            kind: &layer.kind,
            texture_index: self.layer_texture_index(layer.depth),
            bbox,
            sample,
            target: None,
        }
    }

    fn open_root_layer(&self) -> OpenLayer<'a> {
        OpenLayer {
            cmds: &self.recorder.root_cmds,
            kind: &REGULAR_LAYER_KIND,
            texture_index: TextureIndex::Odd,
            bbox: self.scene_bbox,
            sample: LayerSamplePlacement::regular(self.scene_bbox),
            target: None,
        }
    }

    fn layer_texture_index(&self, layer_depth: usize) -> TextureIndex {
        TextureIndex::from_parity(layer_depth + usize::from(self.recorder.root_is_blend_target))
    }

    fn push_draws<T: ScheduleTarget>(
        &mut self,
        draws: &core::ops::Range<u32>,
        state: &mut TargetScheduleState<T>,
        rounds: &mut Rounds,
    ) where
        T: ScheduleTarget,
    {
        if draws.is_empty() {
            return;
        }

        rounds.build_draw(state, &mut self.storage.buffers.draw_buffers, |builder| {
            for draw in &self.recorder.draws[draws.start as usize..draws.end as usize] {
                builder.push_draw(draw, self.strip_storage, self.paint_resolver);
            }
        });
    }

    fn schedule_layer(
        &mut self,
        mut layer: OpenLayer<'a>,
        rounds: &mut Rounds,
    ) -> Result<ScheduledLayer, RenderError> {
        // Overall we follow a similar flow to `schedule_root` here.

        for cmd in layer.cmds {
            // First make sure that the child node is scheduled, in case it exists.
            let child = self.prepare_node(cmd, layer.sample.bbox, rounds)?;

            // This is probably one of the most crucial lines in this scheduling algorithm: As can
            // be seen, when traversing the render graph, we only allocate space for the current
            // layer lazily **after** we have scheduled any potential child node (which happens
            // in the line above), not before. So allocations of layers happens in a bottom-up
            // fashion instead up top-down.
            //
            // This is crucial for memory reasons: Imagine if we had a render graph with 10 nested
            // layers. If we reserved space up eagerly top-down, at peak we would need to reserve
            // space for all 10 layers in the layer texture atlas. On the other hand, by doing
            // bottom-up, we need to retain 2 layers at most if we want to be memory-efficient:
            // Once the child layer has been composed into the parent, it's atlas allocation can
            // be released and therefore the paren't parent can reuse that same space in the next round.
            // In the best case, if we have many small layers, we can still batch many layers
            // in the same round, which is also what we currently do.
            let target = self.ensure_layer_target(&mut layer)?;

            // Now schedule the draws + optional layer composition of this node.
            self.schedule_node(cmd, child, &mut target.schedule_state, rounds);
        }

        self.ensure_layer_target(&mut layer)?;
        let target = layer.target.take().unwrap();
        let region = target.schedule_state.draw_state.target;
        let base_round = target.schedule_state.base_round;
        if let Some(filter) = target.filter {
            let allocation_filter = target.allocations.scratch_allocations;
            rounds.ensure_exists(base_round);
            rounds.rounds[base_round].push_filter_op(
                region.texture.texture_index,
                &mut self.storage.buffers,
                FilterOp {
                    layer_region: region,
                    scratches: allocation_filter
                        .map(|scratch| scratch.map(|texture| texture.region)),
                    filter_data_offset: filter.data_offset,
                    gpu_filter: filter.data,
                },
            );
        }
        let scheduled = ScheduledLayer {
            sample: layer.sample.resolve(region),
            allocations: target.allocations,
            ready_round: base_round,
        };
        self.unreleased_layer_count += 1;

        Ok(scheduled)
    }

    /// Schedule a composition operation for an arbitrary layer.
    fn compose_layer(
        &mut self,
        props: &LayerProps,
        child_layer: ScheduledLayer,
        state: &mut TargetScheduleState<LayerTextureRegion>,
        rounds: &mut Rounds,
    ) {
        let blend_mode = props.blend_mode;
        let opacity = props.opacity;
        let child_texture_index = child_layer.sample.source.texture.texture_index;

        if blend_mode == BlendMode::default() {
            self.compose_simple_layer(props, child_layer, state, rounds);

            return;
        }

        let parent_region = state.draw_state.target;
        let source_bbox = child_layer.sample.bbox;
        let affected_bbox = if blend_mode.is_destructive() {
            let parent_bbox = parent_region.layer_bbox;

            props
                .clip_path
                .as_ref()
                .map_or(parent_bbox, |clip| parent_bbox.intersect(clip.bbox))
        } else {
            source_bbox
        };
        let parent_texture_index = parent_region.texture.texture_index;
        debug_assert_ne!(
            parent_texture_index, child_texture_index,
            "blended parent and child layers must use opposite textures"
        );
        let blend_round = state.base_round.max(
            state
                .draw_state
                .target
                .min_round(child_texture_index, child_layer.ready_round),
        );
        let bbox = affected_bbox.intersect(parent_region.layer_bbox);
        if bbox.is_empty() {
            self.release_layer(child_layer, blend_round, rounds);
            state.base_round = state.base_round.max(blend_round);
            return;
        }

        rounds.ensure_exists(blend_round);
        rounds.rounds[blend_round].scratch_texture_clears[BLEND_SCRATCH_INDEX.get_index()]
            .push(parent_region.blend_scratch_clear_rect(bbox));
        rounds.rounds[blend_round].push_blend_op(
            parent_texture_index,
            &mut self.storage.buffers,
            BlendOp {
                parent_region,
                child_region: child_layer.sample.source,
                blend_bbox: bbox,
                blend_mode,
                opacity,
            },
        );

        self.release_layer(child_layer, blend_round, rounds);

        state.base_round = blend_round + 1;
    }

    /// Schedule a composition operation for a layer using src-over blending.
    fn compose_simple_layer<T: ScheduleTarget>(
        &mut self,
        props: &LayerProps,
        child_layer: ScheduledLayer,
        state: &mut TargetScheduleState<T>,
        rounds: &mut Rounds,
    ) {
        let child_texture_index = child_layer.sample.source.texture.texture_index;

        // Layer invocations can introduce a barrier! We need to update the base round of the
        // current target such that the layer fill is scheduled only once the layer actually
        // finished rendering and is available to us for sampling. All future draws should also
        // only be scheduled at that new base round, or later.
        state.base_round = state.base_round.max(
            state
                .draw_state
                .target
                .min_round(child_texture_index, child_layer.ready_round),
        );

        // Schedule the actual layer fill command.
        rounds.build_draw(state, &mut self.storage.buffers.draw_buffers, |builder| {
            builder.push_layer_fill(
                child_layer.sample,
                props.opacity,
                props.clip_path.as_ref(),
                self.strip_storage,
            );
        });

        // Now that the child layer has been composited into the parent, don't forget to release
        // the child layer at the end of this round, since its rendered representation does not
        // need to be retained in the layer texture anymore!
        self.release_layer(child_layer, state.base_round, rounds);
    }

    fn release_layer(&mut self, layer: ScheduledLayer, round_idx: usize, rounds: &mut Rounds) {
        // When releasing the layer, we need to make sure whatever regions in the different textures
        // where used are cleared properly.

        self.unreleased_layer_count -= 1;
        rounds.ensure_exists(round_idx);

        // First the main layer allocation.
        let layer_region = layer.allocations.main_allocation.clear_region();
        rounds.rounds[round_idx].layer_texture_clears[layer_region.texture_index.get_index()]
            .push(layer_region.rect);

        // Then any potential scratch texture allocations.
        for scratch_region in layer.allocations.scratch_allocations.into_iter().flatten() {
            let clear_region = scratch_region.clear_region();

            rounds.rounds[round_idx].scratch_texture_clears[clear_region.texture_index.get_index()]
                .push(clear_region.rect);
        }

        // And make sure the atlas allocation will also be released.
        self.cursor.release(layer.allocations, round_idx);
    }

    /// Lazily allocate space for an open layer.
    fn ensure_layer_target<'b>(
        &mut self,
        layer: &'b mut OpenLayer<'a>,
    ) -> Result<&'b mut LayerTarget, RenderError> {
        if layer.target.is_none() {
            let filter = match layer.kind {
                RecordedLayerKind::Filter { filter_data, .. } => {
                    Some(self.storage.filter_context.push(filter_data))
                }
                RecordedLayerKind::Regular => None,
            };

            let request = LayerAllocationRequest::new(layer, filter.as_ref());
            // Note: this might advance the base round, in case the atlas is already full
            // and we therefore need to advance the round cursor until enough space has been
            // freed.
            let allocation = self.cursor.allocate(request)?;
            let region = LayerTextureRegion {
                texture: allocation.allocation.main_allocation.region,
                layer_bbox: layer.bbox,
            };

            let schedule_state = TargetScheduleState::new_layer(region, allocation.round_idx);
            layer.target = Some(LayerTarget {
                allocations: allocation.allocation,
                filter,
                schedule_state,
            });
        }

        Ok(layer.target.as_mut().unwrap())
    }
}

/// A layer that has been scheduled and can be sampled by its parent.
#[must_use = "scheduled layers must be released"]
#[derive(Debug)]
struct ScheduledLayer {
    allocations: LayerAllocations,
    sample: LayerSample,
    ready_round: usize,
}

#[derive(Debug)]
struct PreparedChild<'a> {
    props: &'a LayerProps,
    layer: ScheduledLayer,
}

#[derive(Debug)]
struct OpenLayer<'a> {
    cmds: &'a [CmdNode],
    kind: &'a RecordedLayerKind,
    texture_index: TextureIndex,
    bbox: RectU16,
    sample: LayerSamplePlacement,
    target: Option<LayerTarget>,
}

#[derive(Debug, Clone, Copy)]
struct LayerSamplePlacement {
    src_offset: (u16, u16),
    bbox: RectU16,
}

impl LayerSamplePlacement {
    fn regular(bbox: RectU16) -> Self {
        Self {
            src_offset: (0, 0),
            bbox,
        }
    }

    fn filter(placement: FilterLayerPlacement) -> Self {
        Self {
            src_offset: (placement.src_x, placement.src_y),
            bbox: placement.dest_bbox,
        }
    }

    fn resolve(self, allocation: LayerTextureRegion) -> LayerSample {
        let x0 = allocation.texture.rect.x0 + self.src_offset.0;
        let y0 = allocation.texture.rect.y0 + self.src_offset.1;
        LayerSample {
            source: LayerTextureRegion {
                texture: TextureRegion {
                    texture_index: allocation.texture.texture_index,
                    rect: RectU16::new(x0, y0, x0 + self.bbox.width(), y0 + self.bbox.height()),
                },
                layer_bbox: self.bbox,
            },
            bbox: self.bbox,
        }
    }
}

#[derive(Debug)]
struct LayerTarget {
    allocations: LayerAllocations,
    filter: Option<PreparedGpuFilter>,
    schedule_state: TargetScheduleState<LayerTextureRegion>,
}

impl Rounds {
    fn build_draw<T: ScheduleTarget>(
        &mut self,
        state: &mut TargetScheduleState<T>,
        draw_buffers: &mut DrawBuffers,
        f: impl FnOnce(&mut DrawBuilder<'_, T>),
    ) {
        self.ensure_exists(state.base_round);

        let target_draw = state
            .draw_state
            .target
            .draw_mut(&mut self.rounds[state.base_round]);

        let mut builder = DrawBuilder::new(target_draw, draw_buffers, &mut state.draw_state);
        f(&mut builder);
    }
}

#[derive(Debug, Default)]
pub(crate) struct ScheduleBuffers {
    pub(crate) draw_buffers: DrawBuffers,
    pub(crate) filter_ops: Vec<FilterOp>,
    pub(crate) blend_ops: Vec<BlendOp>,
}

impl ScheduleBuffers {
    fn clear(&mut self) {
        self.draw_buffers.clear();
        self.filter_ops.clear();
        self.blend_ops.clear();
    }
}

/// Persistent buffers used to build schedules across frames.
#[derive(Debug, Default)]
pub(crate) struct ScheduleStorage {
    pub(crate) buffers: ScheduleBuffers,
    pub(crate) filter_context: FilterContext,
    filter_pass_plan: FilterPassPlan,
}

impl ScheduleStorage {
    fn clear(&mut self) {
        self.buffers.clear();
        self.filter_context.clear();
    }
}

/// State for scheduling draws to a specific target.
#[derive(Debug)]
struct TargetScheduleState<T: ScheduleTarget> {
    /// The underlying draw state.
    draw_state: DrawState<T>,
    /// The base round at which subsequent draws to this target should be scheduled.
    ///
    /// This value can be incremented when "barriers" are introduced in the render graph.
    /// For example, if the current round is 2 but the rendered contents of a subsequent
    /// layer are only available at round 4, then this value will be updated to 4 after the
    /// layer fill has been generated, such that subsequent draws are also scheduled for round
    /// 4 (or later), ensuring they are applied in the correct order.
    base_round: usize,
}

impl<T: ScheduleTarget> TargetScheduleState<T> {
    fn new(target: T, base_round: usize, target_bbox: RectU16) -> Self {
        Self {
            draw_state: DrawState::new(target, target_bbox),
            base_round,
        }
    }
}

impl TargetScheduleState<LayerTextureRegion> {
    fn new_layer(target: LayerTextureRegion, base_round: usize) -> Self {
        Self::new(target, base_round, target.layer_bbox)
    }
}

trait ScheduleTarget: DrawTarget {
    fn draw_mut<'a>(&self, round: &'a mut Round) -> &'a mut Draw;

    /// Given that a child layer finishes rendering at `child_round` and is stored in
    /// the texture at index `child_texture_index`, return the round smallest round in which
    /// the composition into the parent can be scheduled.
    fn min_round(&self, child_texture_index: TextureIndex, child_round: usize) -> usize;
}

impl ScheduleTarget for RootRenderTarget {
    fn draw_mut<'a>(&self, round: &'a mut Round) -> &'a mut Draw {
        round.root_draw_mut()
    }

    fn min_round(&self, _: TextureIndex, child_round: usize) -> usize {
        // Within a single round, root draws are performed once all layer texture operations
        // finished (see `Rounds::execute`), so we can always schedule it in that same round.
        child_round
    }
}

impl ScheduleTarget for LayerTextureRegion {
    fn draw_mut<'a>(&self, round: &'a mut Round) -> &'a mut Draw {
        round.layer_draw_mut(self.texture.texture_index)
    }

    fn min_round(&self, child_texture_index: TextureIndex, child_round: usize) -> usize {
        // As seen in `Rounds::execute`, the order within a round is:
        // - All even texture draws.
        // - All odd texture draws.
        // - All root draws.
        //
        // Therefore:
        // - If the child is in the even texture, we can schedule the compositing operation in the
        //   same round.
        // - Otherwise, we need to do it in the next round.
        child_round + usize::from(!child_texture_index.is_even())
    }
}
