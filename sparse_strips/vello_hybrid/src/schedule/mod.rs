// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Builds and executes dependency-ordered rendering rounds for `vello_hybrid`.

mod allocate;
mod cursor;
pub(crate) mod execute;
pub(crate) mod round;

use self::allocate::{Allocation, Atlases, LayerAllocationRequest, LayerAllocations};
use self::cursor::Cursor;
pub(crate) use self::execute::{RendererBackend, execute};
use self::round::{BlendOp, FilterOp, Rounds};
use crate::blend::BLEND_SCRATCH_INDEX;
use crate::draw::{DrawBuffers, DrawBuilder, DrawState, LayerSample};
use crate::filter::{FilterContext, FilterPassPlan, PreparedGpuFilter};
use crate::paint::PaintResolver;
use crate::scene::RecordedDraw;
use crate::target::{
    DrawTarget, IntermediateTextureSizes, LayerTextureRegion, RenderTarget, RootRenderTarget,
    TextureRegion,
};
use crate::util::Int16Size;
use crate::{RenderError, Scene};
use alloc::vec::Vec;
use vello_common::geometry::RectU16;
use vello_common::multi_atlas::AtlasError;
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
        filter_context: &mut FilterContext,
        texture_sizes: IntermediateTextureSizes,
    ) -> Result<Self, RenderError> {
        let strip_storage = scene.strip_storage.borrow();
        let scene_bbox = RectU16::new(0, 0, scene.width, scene.height).snap_to_tile_coordinates();
        filter_context.clear();
        storage.buffers.clear();
        let planner = SchedulePlanner::new(
            &scene.recorder,
            scene_bbox,
            &strip_storage,
            root_output_target,
            paint_resolver,
            filter_context,
            texture_sizes,
            storage,
        );

        planner.build()
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
    filter_pass_plan: FilterPassPlan,
}

impl RenderTarget<LayerTextureRegion> {
    fn texture_index(self) -> Option<u8> {
        match self {
            Self::Root(_) => None,
            Self::Layer(region) => Some(region.texture.texture_index),
        }
    }

    fn layer_region(self) -> LayerTextureRegion {
        match self {
            Self::Layer(region) => region,
            Self::Root(_) => panic!("root targets do not have layer regions"),
        }
    }

    fn required_round_for_layer_sample(self, child_texture_index: u8, child_round: usize) -> usize {
        match self.texture_index() {
            Some(parent_texture_index)
                if parent_texture_index != child_texture_index
                    && Self::layer_texture_order(child_texture_index)
                        < Self::layer_texture_order(parent_texture_index) =>
            {
                child_round
            }
            Some(_) => child_round + 1,
            None => child_round,
        }
    }

    fn layer_texture_order(texture_index: u8) -> u8 {
        match texture_index {
            1 => 0,
            0 => 1,
            _ => texture_index,
        }
    }
}

/// Plans concrete, executable rounds from a recorded scene.
#[derive(Debug)]
struct SchedulePlanner<'a, 'p> {
    recorder: &'a CommandRecorder<RecordedDraw>,
    scene_bbox: RectU16,
    strip_storage: &'a StripStorage,
    root_render_target: RootRenderTarget,
    paint_resolver: PaintResolver<'a>,
    cursor: Cursor<Atlases>,
    filter_context: &'p mut FilterContext,
    texture_sizes: IntermediateTextureSizes,
    storage: &'p mut ScheduleStorage,
}

impl<'a, 'p> SchedulePlanner<'a, 'p> {
    fn new(
        recorder: &'a CommandRecorder<RecordedDraw>,
        scene_bbox: RectU16,
        strip_storage: &'a StripStorage,
        root_render_target: RootRenderTarget,
        paint_resolver: PaintResolver<'a>,
        filter_context: &'p mut FilterContext,
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
            filter_context,
            texture_sizes,
            storage,
        }
    }

    fn build(mut self) -> Result<Schedule, RenderError> {
        let mut rounds = Rounds::default();
        rounds.ensure_exists(0);

        let result = self.schedule_root(&mut rounds);

        if let Err(error) = result {
            self.storage.buffers.clear();

            return Err(error);
        }
        self.storage.buffers.draw_buffers.opaque_strips.reverse();

        Ok(Schedule {
            rounds,
            texture_sizes: self.texture_sizes,
        })
    }

    fn schedule_root(&mut self, rounds: &mut Rounds) -> Result<(), RenderError> {
        let ready_round = if self.recorder.root_is_blend_target {
            let layer = self.finish_layer(self.open_root_layer(), rounds)?;

            let target = DrawTarget::Root(self.root_render_target);
            let mut state = DrawState::new(target, layer.ready_round, self.scene_bbox);
            rounds.build_draw(
                &mut state,
                &mut self.storage.buffers.draw_buffers,
                |builder| {
                    builder.push_layer_fill(layer.sample, 1.0, None, self.strip_storage);
                },
            );
            self.release_layer(layer, layer.ready_round, rounds);

            layer.ready_round
        } else {
            let target = DrawTarget::Root(self.root_render_target);
            let mut state = DrawState::new(target, self.cursor.current_round(), self.scene_bbox);
            self.schedule_nodes(&self.recorder.root_cmds, &mut state, rounds)?;
            state.draw_round
        };
        rounds.ensure_exists(ready_round);

        Ok(())
    }

    fn schedule_nodes(
        &mut self,
        cmds: &[CmdNode],
        state: &mut DrawState,
        rounds: &mut Rounds,
    ) -> Result<(), RenderError> {
        for cmd in cmds {
            let child = self.prepare_node(cmd, state.target_bbox, rounds)?;

            self.emit_node(cmd, child, state, rounds);
        }

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

        let scheduled = self.finish_layer(self.open_layer(layer, bbox), rounds)?;
        Ok(Some(PreparedChild {
            props: &layer.props,
            layer: scheduled,
        }))
    }

    fn emit_node(
        &mut self,
        cmd: &CmdNode,
        child: Option<PreparedChild<'a>>,
        state: &mut DrawState,
        rounds: &mut Rounds,
    ) {
        self.push_draws(&cmd.draws, state, rounds);
        if let Some(child) = child {
            self.compose_layer(child.props, child.layer, state, rounds);
        }
    }

    fn open_layer(&self, layer: &'a RecordedLayer, bbox: RectU16) -> OpenLayer<'a> {
        let sample = match &layer.kind {
            RecordedLayerKind::Regular => LayerSamplePlacement {
                src_offset: (0, 0),
                bbox,
            },
            RecordedLayerKind::Filter { placement, .. } => LayerSamplePlacement {
                src_offset: (placement.src_x, placement.src_y),
                bbox: placement.dest_bbox,
            },
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
            texture_index: 1,
            bbox: self.scene_bbox,
            sample: LayerSamplePlacement {
                src_offset: (0, 0),
                bbox: self.scene_bbox,
            },
            target: None,
        }
    }

    fn layer_texture_index(&self, layer_depth: usize) -> u8 {
        ((layer_depth + usize::from(self.recorder.root_is_blend_target)) & 1)
            .try_into()
            .unwrap()
    }

    fn push_draws(
        &mut self,
        draws: &core::ops::Range<u32>,
        state: &mut DrawState,
        rounds: &mut Rounds,
    ) {
        if draws.is_empty() {
            return;
        }

        rounds.build_draw(state, &mut self.storage.buffers.draw_buffers, |builder| {
            for draw in &self.recorder.draws[draws.start as usize..draws.end as usize] {
                builder.push_draw(draw, self.strip_storage, self.paint_resolver);
            }
        });
    }

    fn finish_layer(
        &mut self,
        mut layer: OpenLayer<'a>,
        rounds: &mut Rounds,
    ) -> Result<ScheduledLayer, RenderError> {
        for cmd in layer.cmds {
            let child = self.prepare_node(cmd, layer.sample.bbox, rounds)?;
            if cmd.draws.is_empty() && child.is_none() {
                continue;
            }

            let target = self.ensure_layer_target(&mut layer)?;
            self.emit_node(cmd, child, &mut target.draw_state, rounds);
        }

        self.ensure_layer_target(&mut layer)?;
        let target = layer
            .target
            .take()
            .expect("finished layers must have an allocated target");
        let ready_round = target.draw_state.draw_round;
        if let Some(filter) = target.filter {
            let allocation_filter = target.allocations.scratch_allocations;
            rounds.ensure_exists(ready_round);
            rounds.rounds[ready_round].push_filter_op(
                target.region.texture.texture_index,
                &mut self.storage.buffers,
                FilterOp {
                    layer_region: target.region,
                    scratches: allocation_filter
                        .map(|scratch| scratch.map(|texture| texture.region)),
                    filter_data_offset: filter.data_offset,
                    gpu_filter: filter.data,
                },
            );
        }
        Ok(ScheduledLayer {
            sample: layer.sample.resolve(target.region),
            allocations: target.allocations,
            ready_round,
        })
    }

    fn compose_layer(
        &mut self,
        props: &LayerProps,
        layer: ScheduledLayer,
        state: &mut DrawState,
        rounds: &mut Rounds,
    ) {
        let blend_mode = props.blend_mode;
        let opacity = props.opacity;
        let child_texture_index = layer.sample.source.texture.texture_index;
        if blend_mode == BlendMode::default() {
            debug_assert_ne!(
                state.target.texture_index(),
                Some(child_texture_index),
                "parent and child layers must use opposite textures"
            );
            state.draw_round = state.draw_round.max(
                state
                    .target
                    .required_round_for_layer_sample(child_texture_index, layer.ready_round),
            );
            rounds.build_draw(state, &mut self.storage.buffers.draw_buffers, |builder| {
                builder.push_layer_fill(
                    layer.sample,
                    props.opacity,
                    props.clip_path.as_ref(),
                    self.strip_storage,
                );
            });
            self.release_layer(layer, state.draw_round, rounds);
            return;
        }

        let source_bbox = layer.sample.bbox;
        let affected_bbox = if blend_mode.is_destructive() {
            let parent_bbox = state.target.layer_region().layer_bbox;
            props
                .clip_path
                .as_ref()
                .map_or(parent_bbox, |clip| parent_bbox.intersect(clip.bbox))
        } else {
            source_bbox
        };
        let parent_region = state.target.layer_region();
        let parent_texture_index = parent_region.texture.texture_index;
        debug_assert_ne!(
            parent_texture_index, child_texture_index,
            "blended parent and child layers must use opposite textures"
        );
        let blend_round = state.draw_round.max(
            state
                .target
                .required_round_for_layer_sample(child_texture_index, layer.ready_round),
        );
        let bbox = affected_bbox.intersect(state.target.layer_region().layer_bbox);
        if bbox.is_empty() {
            self.release_layer(layer, blend_round, rounds);
            state.draw_round = state.draw_round.max(blend_round);
            return;
        }

        rounds.ensure_exists(blend_round);
        rounds.rounds[blend_round].scratch_texture_clears[usize::from(BLEND_SCRATCH_INDEX)]
            .push(parent_region.blend_scratch_clear_rect(bbox));
        rounds.rounds[blend_round].push_blend_op(
            parent_texture_index,
            &mut self.storage.buffers,
            BlendOp {
                parent_region,
                child_region: layer.sample.source,
                blend_bbox: bbox,
                blend_mode,
                opacity,
            },
        );
        self.release_layer(layer, blend_round, rounds);
        state.draw_round = blend_round + 1;
    }

    fn release_layer(&mut self, layer: ScheduledLayer, round_idx: usize, rounds: &mut Rounds) {
        rounds.ensure_exists(round_idx);
        let clear_region = layer.allocations.main_allocation.clear_region();
        rounds.rounds[round_idx].layer_texture_clears[usize::from(clear_region.texture_index)]
            .push(clear_region.rect);
        for scratch in layer.allocations.scratch_allocations.into_iter().flatten() {
            let clear_region = scratch.clear_region();
            rounds.rounds[round_idx].scratch_texture_clears
                [usize::from(clear_region.texture_index)]
            .push(clear_region.rect);
        }
        self.cursor.release(layer.allocations, round_idx);
    }

    fn allocate_region(
        &mut self,
        texture_index: u8,
        bbox: RectU16,
        kind: &RecordedLayerKind,
        scratch_count: u8,
    ) -> Result<(Allocation<LayerAllocations>, LayerTextureRegion), RenderError> {
        let request = LayerAllocationRequest::new(
            texture_index,
            Int16Size::new(bbox.width(), bbox.height()),
            kind,
            scratch_count,
        );

        let Some(allocation) = self.cursor.allocate(request) else {
            return Err(RenderError::AtlasError(AtlasError::NoSpaceAvailable));
        };

        let region = LayerTextureRegion {
            texture: allocation.allocation.main_allocation.region,
            layer_bbox: bbox,
        };

        Ok((allocation, region))
    }

    fn ensure_layer_target<'b>(
        &mut self,
        layer: &'b mut OpenLayer<'a>,
    ) -> Result<&'b mut LayerTarget, RenderError> {
        if layer.target.is_none() {
            let filter = match layer.kind {
                RecordedLayerKind::Filter { filter_data, .. } => {
                    Some(self.filter_context.push(filter_data))
                }
                RecordedLayerKind::Regular => None,
            };
            let (allocation, region) = self.allocate_region(
                layer.texture_index,
                layer.bbox,
                layer.kind,
                filter.map_or(0, PreparedGpuFilter::scratch_count),
            )?;
            let draw_state = DrawState::new(
                DrawTarget::Layer(region),
                allocation.round_idx,
                region.layer_bbox,
            );
            layer.target = Some(LayerTarget {
                allocations: allocation.allocation,
                region,
                filter,
                draw_state,
            });
        }

        Ok(layer
            .target
            .as_mut()
            .expect("layer target must be initialized"))
    }
}

/// A layer that has been scheduled and can be sampled by its parent.
#[derive(Debug, Clone, Copy)]
struct ScheduledLayer {
    allocations: LayerAllocations,
    sample: LayerSample,
    ready_round: usize,
}

#[derive(Debug, Clone, Copy)]
struct PreparedChild<'a> {
    props: &'a LayerProps,
    layer: ScheduledLayer,
}

#[derive(Debug)]
struct OpenLayer<'a> {
    cmds: &'a [CmdNode],
    kind: &'a RecordedLayerKind,
    texture_index: u8,
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
    region: LayerTextureRegion,
    filter: Option<PreparedGpuFilter>,
    draw_state: DrawState,
}

impl Rounds {
    fn build_draw(
        &mut self,
        state: &mut DrawState,
        draw_buffers: &mut DrawBuffers,
        f: impl FnOnce(&mut DrawBuilder<'_>),
    ) {
        self.ensure_exists(state.draw_round);

        let target_draw = match state.target {
            DrawTarget::Root(_) => self.rounds[state.draw_round].root_draw_mut(),
            DrawTarget::Layer(region) => {
                self.rounds[state.draw_round].layer_draw_mut(region.texture.texture_index)
            }
        };

        let mut builder = DrawBuilder::new(target_draw, draw_buffers, state);
        f(&mut builder);
    }
}
