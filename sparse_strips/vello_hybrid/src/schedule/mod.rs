// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Builds and executes dependency-ordered rendering rounds for `vello_hybrid`.

mod allocate;
mod cursor;
mod draw;
pub(crate) mod execute;
pub(crate) mod round;

use self::allocate::{Allocation, Atlases, LayerAllocationRequest, LayerAllocations};
use self::cursor::Cursor;
use self::draw::{DepthCounter, DrawBuilder, LayerSample};
pub(crate) use self::execute::{RendererBackend, execute};
use self::round::{BlendOp, FilterOp, Rounds};
use crate::blend::BLEND_SCRATCH_INDEX;
use crate::filter::{FilterContext, FilterPassPlan, PreparedGpuFilter};
use crate::paint::PaintResolver;
use crate::target::{
    DrawTarget, IntermediateTextureSizes, LayerTextureRegion, RenderTarget, RootRenderTarget,
    TextureRegion,
};
use crate::util::Int16Size;
use crate::{GpuStrip, RenderError, Scene};
use alloc::vec::Vec;
use vello_common::TextureId;
use vello_common::geometry::RectU16;
use vello_common::multi_atlas::AtlasError;
use vello_common::peniko::BlendMode;
use vello_common::record::{CmdNode, RecordedLayerKind};
use vello_common::strip_generator::StripStorage;
use vello_common::util::RectExt;

const REGULAR_LAYER_KIND: RecordedLayerKind = RecordedLayerKind::Regular;

/// Specifies a run of strips inside a draw that can be drawn with the same external texture
/// binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExternalTextureRun {
    pub(crate) texture_id: TextureId,
    /// Start index of the strip range for this run. The end is implicitly the start of the next
    /// run, or, for the last run, the total number of strips.
    pub(crate) strips_start: usize,
}

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
        filter_context.clear();
        storage.buffers.clear();
        let planner = SchedulePlanner::new(
            scene,
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
    pub(crate) opaque_strips: Vec<GpuStrip>,
    pub(crate) strips: Vec<GpuStrip>,
    pub(crate) filter_ops: Vec<FilterOp>,
    pub(crate) blend_ops: Vec<BlendOp>,
}

impl ScheduleBuffers {
    fn clear(&mut self) {
        self.opaque_strips.clear();
        self.strips.clear();
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

    fn geometry_shift(self) -> (i32, i32) {
        match self {
            Self::Root(_) => (0, 0),
            Self::Layer(region) => region.geometry_shift(),
        }
    }

    fn draw_bounds(self, scene: &Scene) -> RectU16 {
        match self {
            Self::Root(_) => {
                RectU16::new(0, 0, scene.width, scene.height).snap_to_tile_coordinates()
            }
            Self::Layer(region) => region.scene_bbox,
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
    scene: &'a Scene,
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
        scene: &'a Scene,
        strip_storage: &'a StripStorage,
        root_render_target: RootRenderTarget,
        paint_resolver: PaintResolver<'a>,
        filter_context: &'p mut FilterContext,
        texture_sizes: IntermediateTextureSizes,
        storage: &'p mut ScheduleStorage,
    ) -> Self {
        Self {
            scene,
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
        self.storage.buffers.opaque_strips.reverse();

        Ok(Schedule {
            rounds,
            texture_sizes: self.texture_sizes,
        })
    }

    fn schedule_root(&mut self, rounds: &mut Rounds) -> Result<(), RenderError> {
        let ready_round = if self.scene.recorder.root_is_blend_target {
            self.schedule_root_blend_target(rounds)?
        } else {
            self.schedule_root_direct(rounds)?
        };
        rounds.ensure_exists(ready_round);

        Ok(())
    }

    fn schedule_root_direct(&mut self, rounds: &mut Rounds) -> Result<usize, RenderError> {
        let target = DrawTarget::Root(self.root_render_target);
        let mut state = DrawState::new(
            target,
            self.cursor.current_round(),
            target.draw_bounds(self.scene),
        );
        let ready_round =
            self.schedule_commands(&self.scene.recorder.root_cmds, &mut state, rounds);
        ready_round
    }

    fn schedule_root_blend_target(&mut self, rounds: &mut Rounds) -> Result<usize, RenderError> {
        let layer = self.finish_layer(self.push_root_layer(), rounds)?;

        let target = DrawTarget::Root(self.root_render_target);
        let mut state = DrawState::new(target, layer.round_idx, target.draw_bounds(self.scene));
        rounds.with_draw_builder(&mut state, &mut self.storage.buffers, |builder| {
            builder.push_layer_fill(layer.sample, 1.0, None, self.strip_storage);
        });
        self.release_layer(layer, layer.round_idx, rounds);

        Ok(layer.round_idx)
    }

    fn schedule_commands(
        &mut self,
        cmds: &[CmdNode],
        state: &mut DrawState,
        rounds: &mut Rounds,
    ) -> Result<usize, RenderError> {
        for cmd in cmds {
            if let Some(layer_id) = cmd.layer {
                let Some(bbox) = self.layer_bbox(layer_id, state.draw_bounds) else {
                    self.push_draws(&cmd.draws, state, rounds);
                    continue;
                };
                let child_layer = self.schedule_layer_subtree(layer_id, bbox, rounds)?;
                self.push_draws(&cmd.draws, state, rounds);
                self.schedule_child_layer(layer_id, child_layer, state, rounds);
            } else {
                self.push_draws(&cmd.draws, state, rounds);
            }
        }

        Ok(state.round_idx)
    }

    fn schedule_layer_subtree(
        &mut self,
        layer_id: u32,
        bbox: RectU16,
        rounds: &mut Rounds,
    ) -> Result<ScheduledLayer, RenderError> {
        let layer = self.push_layer(layer_id, bbox);
        let layer = self.finish_layer(layer, rounds)?;
        Ok(layer)
    }

    fn layer_bbox(&self, layer_id: u32, parent_bbox: RectU16) -> Option<RectU16> {
        let layer = &self.scene.recorder.layers[layer_id as usize];
        let bbox = if layer.bbox.is_empty() {
            if !layer.props.blend_mode.is_destructive() {
                return None;
            }

            layer
                .props
                .clip_path
                .as_ref()
                .map_or(parent_bbox, |clip| parent_bbox.intersect(clip.bbox))
        } else {
            layer.bbox
        };
        (!bbox.is_empty()).then_some(bbox)
    }

    fn push_layer(&self, layer_id: u32, bbox: RectU16) -> OpenLayer<'a> {
        let layer = &self.scene.recorder.layers[layer_id as usize];
        let allocation_bbox = bbox.snap_to_tile_coordinates();
        let sample = if layer.bbox.is_empty() {
            LayerSamplePlacement {
                source_offset: (0, 0),
                source_scene_bbox: allocation_bbox,
                bbox,
            }
        } else {
            match &layer.kind {
                RecordedLayerKind::Regular => LayerSamplePlacement {
                    source_offset: (0, 0),
                    source_scene_bbox: allocation_bbox,
                    bbox,
                },
                RecordedLayerKind::Filter { placement, .. } => LayerSamplePlacement {
                    source_offset: (placement.src_x, placement.src_y),
                    source_scene_bbox: placement.dest_bbox,
                    bbox: placement.dest_bbox,
                },
            }
        };

        OpenLayer {
            cmds: &layer.cmds,
            kind: &layer.kind,
            texture_index: self.layer_texture_index(layer.depth),
            allocation_bbox,
            sample,
            target: None,
        }
    }

    fn push_root_layer(&self) -> OpenLayer<'a> {
        let bbox = DrawTarget::Root(self.root_render_target).draw_bounds(self.scene);

        OpenLayer {
            cmds: &self.scene.recorder.root_cmds,
            kind: &REGULAR_LAYER_KIND,
            texture_index: 1,
            allocation_bbox: bbox,
            sample: LayerSamplePlacement {
                source_offset: (0, 0),
                source_scene_bbox: bbox,
                bbox,
            },
            target: None,
        }
    }

    fn layer_texture_index(&self, layer_depth: usize) -> u8 {
        ((layer_depth + usize::from(self.scene.recorder.root_is_blend_target)) & 1)
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

        rounds.with_draw_builder(state, &mut self.storage.buffers, |builder| {
            for draw in &self.scene.recorder.draws[draws.start as usize..draws.end as usize] {
                builder.push_draw(draw, self.strip_storage, self.paint_resolver);
            }
        });
    }

    fn schedule_layer_contents(
        &mut self,
        layer: &mut OpenLayer<'a>,
        rounds: &mut Rounds,
    ) -> Result<(), RenderError> {
        let cmds = layer.cmds;
        for cmd in cmds {
            if let Some(child_layer_id) = cmd.layer {
                let Some(bbox) = self.layer_bbox(child_layer_id, layer.sample.bbox) else {
                    if !cmd.draws.is_empty() {
                        let target = self.layer_target(layer)?;
                        self.push_draws(&cmd.draws, &mut target.draw_state, rounds);
                    }
                    continue;
                };
                let child_layer = self.schedule_layer_subtree(child_layer_id, bbox, rounds)?;

                let target = self.layer_target(layer)?;
                self.push_draws(&cmd.draws, &mut target.draw_state, rounds);
                self.schedule_child_layer(
                    child_layer_id,
                    child_layer,
                    &mut target.draw_state,
                    rounds,
                );
            } else if !cmd.draws.is_empty() {
                let target = self.layer_target(layer)?;
                self.push_draws(&cmd.draws, &mut target.draw_state, rounds);
            }
        }

        Ok(())
    }

    fn finish_layer(
        &mut self,
        mut layer: OpenLayer<'a>,
        rounds: &mut Rounds,
    ) -> Result<ScheduledLayer, RenderError> {
        self.schedule_layer_contents(&mut layer, rounds)?;
        if layer.target.is_none() {
            self.layer_target(&mut layer)?;
        }
        let target = layer
            .target
            .take()
            .expect("finished layers must have an allocated target");
        let ready_round = target.draw_state.round_idx;
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
            region: target.region,
            round_idx: ready_round,
        })
    }

    fn schedule_child_layer(
        &mut self,
        layer_id: u32,
        layer: ScheduledLayer,
        state: &mut DrawState,
        rounds: &mut Rounds,
    ) {
        let props = &self.scene.recorder.layers[layer_id as usize].props;
        let blend_mode = props.blend_mode;
        let opacity = props.opacity;
        let same_texture_as_target =
            state.target.texture_index() == Some(layer.region.texture.texture_index);
        if blend_mode == BlendMode::default() && !same_texture_as_target {
            state.round_idx = state
                .round_idx
                .max(state.target.required_round_for_layer_sample(
                    layer.region.texture.texture_index,
                    layer.round_idx,
                ));
            rounds.with_draw_builder(state, &mut self.storage.buffers, |builder| {
                builder.push_layer_fill(
                    layer.sample,
                    props.opacity,
                    props.clip_path.as_ref(),
                    self.strip_storage,
                );
            });
            self.release_layer(layer, state.round_idx, rounds);
            return;
        }

        let source_bbox = layer.sample.bbox;
        let affected_bbox = if blend_mode.is_destructive() {
            let parent_bbox = state.target.layer_region().scene_bbox;
            props
                .clip_path
                .as_ref()
                .map_or(parent_bbox, |clip| parent_bbox.intersect(clip.bbox))
        } else {
            source_bbox
        };
        let parent_ready_round = state.round_idx;
        let parent_region = state.target.layer_region();
        let parent_texture_index = parent_region.texture.texture_index;
        debug_assert_ne!(
            parent_texture_index, layer.region.texture.texture_index,
            "blended parent and child layers must use opposite textures"
        );
        let blend_round =
            parent_ready_round.max(layer.round_idx + usize::from(parent_texture_index == 1));
        let bbox = affected_bbox.intersect(state.target.layer_region().scene_bbox);
        if bbox.is_empty() {
            self.release_layer(layer, blend_round, rounds);
            state.round_idx = state.round_idx.max(blend_round);
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
        state.round_idx = blend_round + 1;
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
        self.cursor.release_after(layer.allocations, round_idx);
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
            scene_bbox: bbox,
        };

        Ok((allocation, region))
    }

    fn layer_target<'b>(
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
                layer.allocation_bbox,
                layer.kind,
                filter.map_or(0, PreparedGpuFilter::scratch_count),
            )?;
            let draw_state = DrawState::new(
                DrawTarget::Layer(region),
                allocation.round_idx,
                region.scene_bbox,
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
    region: LayerTextureRegion,
    sample: LayerSample,
    round_idx: usize,
}

#[derive(Debug)]
struct OpenLayer<'a> {
    cmds: &'a [CmdNode],
    kind: &'a RecordedLayerKind,
    texture_index: u8,
    allocation_bbox: RectU16,
    sample: LayerSamplePlacement,
    target: Option<LayerTarget>,
}

#[derive(Debug, Clone, Copy)]
struct LayerSamplePlacement {
    source_offset: (u16, u16),
    source_scene_bbox: RectU16,
    bbox: RectU16,
}

impl LayerSamplePlacement {
    fn resolve(self, allocation: LayerTextureRegion) -> LayerSample {
        let x0 = allocation.texture.rect.x0 + self.source_offset.0;
        let y0 = allocation.texture.rect.y0 + self.source_offset.1;
        LayerSample {
            source: LayerTextureRegion {
                texture: TextureRegion {
                    texture_index: allocation.texture.texture_index,
                    rect: RectU16::new(
                        x0,
                        y0,
                        x0 + self.source_scene_bbox.width(),
                        y0 + self.source_scene_bbox.height(),
                    ),
                },
                scene_bbox: self.source_scene_bbox,
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

#[derive(Debug)]
struct DrawState {
    target: DrawTarget,
    depth: DepthCounter,
    draw_bounds: RectU16,
    round_idx: usize,
}

impl DrawState {
    fn new(target: DrawTarget, round_idx: usize, draw_bounds: RectU16) -> Self {
        Self {
            target,
            depth: DepthCounter::default(),
            draw_bounds,
            round_idx,
        }
    }
}

impl Rounds {
    fn with_draw_builder(
        &mut self,
        state: &mut DrawState,
        buffers: &mut ScheduleBuffers,
        f: impl FnOnce(&mut DrawBuilder<'_>),
    ) {
        self.ensure_exists(state.round_idx);

        let target_draw = match state.target {
            DrawTarget::Root(_) => self.rounds[state.round_idx].root_draw_mut(),
            DrawTarget::Layer(region) => {
                self.rounds[state.round_idx].layer_draw_mut(region.texture.texture_index)
            }
        };

        let opaque = state
            .target
            .enable_opaque()
            .then_some(&mut buffers.opaque_strips);
        let mut builder = DrawBuilder::new(target_draw, &mut buffers.strips, opaque, state);
        f(&mut builder);
    }
}
