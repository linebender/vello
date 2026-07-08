// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Round construction for the new hybrid scheduler.

use super::allocate::{Allocation, Atlases, LayerAllocation, LayerAllocationRequest};
use super::buffer::ScheduleBuffers;
use super::cursor::Cursor;
use super::draw::{DepthCounter, DrawBuilder, LayerSample, OpaqueStrips, OpaqueStripsExt};
use super::pool::Pools;
use super::round::{BlendOp, FilterOp, Rounds};
use super::{LayerTextureRegion, RenderTarget, RootRenderTarget, Schedule, TextureRegion};
use crate::blend::BLEND_SCRATCH_INDEX;
use crate::filter::{FilterContext, FilterPlanScratch, PreparedGpuFilter, build_filter_plan};
use crate::paint::Paints;
use crate::scene::RecordedDraw;
use crate::{RenderError, Scene};
use alloc::vec::Vec;
use vello_common::geometry::RectU16;
use vello_common::multi_atlas::AtlasError;
use vello_common::peniko::{BlendMode, Compose};
use vello_common::record::{Drawable, RecordedCmd, RecordedLayerKind};
use vello_common::strip_generator::StripStorage;
use vello_common::util::RectExt;

/// Builds concrete, executable rounds from a recorded scene.
#[derive(Debug)]
pub(super) struct ScheduleBuilder<'a, 'p> {
    scene: &'a Scene,
    strip_storage: &'a StripStorage,
    root_render_target: RootRenderTarget,
    paints: Paints<'a>,
    cursor: Cursor<Atlases>,
    layer_allocations: Vec<Option<ScheduledLayer>>,
    filter_context: &'a FilterContext,
    layer_texture_size: (u32, u32),
    buffers: &'p mut ScheduleBuffers,
    pools: &'p mut Pools,
    filter_plan_scratch: &'p mut FilterPlanScratch,
}

impl<'a, 'p> ScheduleBuilder<'a, 'p> {
    pub(super) fn new(
        scene: &'a Scene,
        strip_storage: &'a StripStorage,
        root_render_target: RootRenderTarget,
        paints: Paints<'a>,
        filter_context: &'a FilterContext,
        layer_texture_size: (u32, u32),
        pools: &'p mut Pools,
        buffers: &'p mut ScheduleBuffers,
        filter_plan_scratch: &'p mut FilterPlanScratch,
    ) -> Self {
        Self {
            scene,
            strip_storage,
            root_render_target,
            paints,
            cursor: Cursor::new(Atlases::new(layer_texture_size)),
            layer_allocations: alloc::vec![None; scene.recorder.layers.len()],
            filter_context,
            layer_texture_size,
            buffers,
            pools,
            filter_plan_scratch,
        }
    }

    pub(super) fn build(&mut self) -> Result<Schedule, RenderError> {
        // TODO: Reuse round storage across frames so large schedules do not repeatedly allocate.
        let mut opaque_strips = None;
        let mut rounds = Rounds {
            rounds: alloc::vec![self.pools.take_round()],
        };

        // Walk the command tree left-to-right. Layer subtrees are scheduled lazily when their
        // parent command stream first needs to sample them. Atlas allocation is monotonic: pressure
        // advances the base round and applies completed cleanups instead of patching old states.
        if let Err(error) = self.schedule_root(&mut opaque_strips, &mut rounds) {
            self.pools.submit_opaque_strips(opaque_strips);
            self.buffers.clear();
            rounds.recycle(self.pools);
            return Err(error);
        }
        opaque_strips.reverse();
        self.build_filter_plans(&mut rounds);

        Ok(Schedule {
            opaque_strips,
            rounds,
        })
    }

    fn schedule_root(
        &mut self,
        opaque_strips: &mut OpaqueStrips,
        rounds: &mut Rounds,
    ) -> Result<(), RenderError> {
        let cmds = &self.scene.recorder.root_cmds;
        if self.scene.recorder.root_is_blend_target {
            let bbox =
                RectU16::new(0, 0, self.scene.width, self.scene.height).snap_to_tile_coordinates();
            if bbox.is_empty() {
                return Ok(());
            }

            let (allocation, region) = self.allocate_region(1, bbox, 0)?;
            let target = RenderTarget::Layer(region);
            let mut state = CommandStreamState::new(
                target,
                allocation.round_idx,
                None,
                target.draw_bounds(self.scene),
            );
            let ready_round = self.schedule_command_stream(cmds, &mut state, rounds)?;

            rounds.ensure_exists(ready_round, self.pools);
            let mut root_state = CommandStreamState::new(
                RenderTarget::Root,
                ready_round,
                None,
                RectU16::new(0, 0, self.scene.width, self.scene.height).snap_to_tile_coordinates(),
            );
            rounds.with_draw_builder(&mut root_state, self.pools, &mut *self.buffers, |builder| {
                builder.push_layer_fill(
                    LayerSample {
                        source: region,
                        bbox: region.scene_bbox,
                        source_origin: (0, 0),
                    },
                    1.0,
                    None,
                    self.strip_storage,
                );
            });
            let clear_region = allocation.allocation.texture.clear_region();
            rounds.rounds[ready_round].layer_texture_clears[clear_region.texture_index]
                .push(clear_region.rect);
            self.cursor
                .release_after(allocation.allocation, ready_round);
        } else {
            let target = RenderTarget::Root;
            let opaque = self
                .pools
                .take_opaque_strips(self.root_render_target == RootRenderTarget::UserSurface);
            let mut state = CommandStreamState::new(
                target,
                self.cursor.current_round(),
                opaque,
                target.draw_bounds(self.scene),
            );
            let ready_round = self.schedule_command_stream(cmds, &mut state, rounds);
            *opaque_strips = state.opaque;
            let ready_round = ready_round?;
            rounds.ensure_exists(ready_round, self.pools);
        }

        Ok(())
    }

    fn build_filter_plans(&mut self, rounds: &mut Rounds) {
        let target_texture_size = (
            u16::try_from(self.layer_texture_size.0)
                .expect("layer texture width must fit into u16"),
            u16::try_from(self.layer_texture_size.1)
                .expect("layer texture height must fit into u16"),
        );
        let ScheduleBuffers {
            filter_ops,
            filter_instances,
            filter_copies,
            ..
        } = &mut *self.buffers;

        for round in &mut rounds.rounds {
            for layer_pass in &mut round.layer_passes {
                let filter_ops = filter_ops.ranged(&layer_pass.filter_ranges);
                build_filter_plan(
                    filter_ops.iter().copied(),
                    target_texture_size,
                    self.filter_plan_scratch,
                );

                layer_pass.filter_passes.steps.clear();
                let step_count = self
                    .filter_plan_scratch
                    .steps
                    .iter()
                    .rposition(|step| !step.is_empty())
                    .map_or(0, |index| index + 1);
                layer_pass.filter_passes.steps.extend(
                    self.filter_plan_scratch.steps[..step_count]
                        .iter()
                        .map(|step| filter_instances.extend_from_slice(step)),
                );
                layer_pass.filter_passes.copy_back =
                    filter_copies.extend_from_slice(&self.filter_plan_scratch.copy_back);
            }
        }
    }

    fn schedule_command_stream(
        &mut self,
        cmds: &[RecordedCmd],
        state: &mut CommandStreamState,
        rounds: &mut Rounds,
    ) -> Result<usize, RenderError> {
        let command_count = cmds.len();
        let mut segment_start = 0;
        for (cmd_idx, cmd) in cmds.iter().enumerate() {
            match cmd {
                RecordedCmd::Draws(_) => {}
                RecordedCmd::Layer(layer_id) => {
                    self.push_command_batches(cmds, segment_start, cmd_idx, state, rounds);

                    let layer_idx = *layer_id as usize;
                    self.schedule_layer_subtree(*layer_id, rounds)?;
                    let Some(layer) = self.layer_allocations[layer_idx] else {
                        let props = &self.scene.recorder.layers[layer_idx].props;
                        if props.blend_mode.is_destructive() {
                            self.schedule_empty_destructive_blend(
                                state,
                                props.blend_mode,
                                props.opacity,
                                rounds,
                            );
                        }
                        segment_start = cmd_idx + 1;
                        continue;
                    };

                    self.schedule_child_layer_sample(*layer_id, layer, state, rounds);
                    segment_start = cmd_idx + 1;
                }
            }
        }

        self.push_command_batches(cmds, segment_start, command_count, state, rounds);
        Ok(self.finish_stream_segment(state, rounds))
    }

    fn schedule_layer_subtree(
        &mut self,
        layer_id: u32,
        rounds: &mut Rounds,
    ) -> Result<(), RenderError> {
        if self.layer_allocations[layer_id as usize].is_some() {
            return Ok(());
        }

        let layer = &self.scene.recorder.layers[layer_id as usize];
        let bbox = layer.bbox;
        if bbox.is_empty() {
            return Ok(());
        }

        let allocation_bbox = bbox.snap_to_tile_coordinates();
        let texture_index = self.layer_texture_index(layer.depth);
        if let Some(layer) =
            self.schedule_layer_command_stream(layer_id, texture_index, allocation_bbox, rounds)?
        {
            self.layer_allocations[layer_id as usize] = Some(layer);
        }

        Ok(())
    }

    fn layer_texture_index(&self, layer_depth: usize) -> usize {
        (layer_depth + usize::from(self.scene.recorder.root_is_blend_target)) & 1
    }

    fn push_command_batches(
        &mut self,
        cmds: &[RecordedCmd],
        start: usize,
        end: usize,
        state: &mut CommandStreamState,
        rounds: &mut Rounds,
    ) {
        let mut draw_ranges = cmds[start..end].iter().filter_map(|cmd| match cmd {
            RecordedCmd::Draws(range) => Some(range.clone()),
            RecordedCmd::Layer(_) => None,
        });
        let Some(first_draw_range) = draw_ranges.next() else {
            return;
        };

        let mut bbox = RectU16::INVERTED;
        rounds.with_draw_builder(state, self.pools, &mut *self.buffers, |builder| {
            for range in core::iter::once(first_draw_range).chain(draw_ranges) {
                for draw in &self.scene.recorder.draws[range.start as usize..range.end as usize] {
                    let strips = match draw {
                        RecordedDraw::Path(path) => &self.strip_storage.strips[path.strips.clone()],
                        RecordedDraw::Rect(_) => &[],
                    };
                    bbox.union(draw.bbox(strips));
                    builder.push_draw(draw, self.strip_storage, self.paints);
                }
            }
        });
        state.backdrop_bbox.union(bbox);
    }

    fn schedule_layer_command_stream(
        &mut self,
        layer_id: u32,
        texture_index: usize,
        bbox: RectU16,
        rounds: &mut Rounds,
    ) -> Result<Option<ScheduledLayer>, RenderError> {
        let layer_idx = layer_id as usize;
        let cmds = &self.scene.recorder.layers[layer_idx].cmds;
        let command_count = cmds.len();
        let mut target = None;
        let mut segment_start = 0;

        for (cmd_idx, cmd) in cmds.iter().enumerate() {
            let child_layer_id = match cmd {
                RecordedCmd::Draws(_) => continue,
                RecordedCmd::Layer(child_layer_id) => *child_layer_id,
            };

            let child_layer_idx = child_layer_id as usize;
            self.schedule_layer_subtree(child_layer_id, rounds)?;
            let Some(child_layer) = self.layer_allocations[child_layer_idx] else {
                let props = &self.scene.recorder.layers[child_layer_idx].props;
                let blend_mode = props.blend_mode;
                let opacity = props.opacity;
                if blend_mode.is_destructive() {
                    let target =
                        self.layer_command_target(layer_id, texture_index, bbox, &mut target)?;
                    self.push_layer_batches(
                        layer_idx,
                        segment_start,
                        cmd_idx,
                        &mut target.stream,
                        rounds,
                    );
                    self.schedule_empty_destructive_blend(
                        &mut target.stream,
                        blend_mode,
                        opacity,
                        rounds,
                    );
                    segment_start = cmd_idx + 1;
                }
                continue;
            };

            let target = self.layer_command_target(layer_id, texture_index, bbox, &mut target)?;
            self.push_layer_batches(
                layer_idx,
                segment_start,
                cmd_idx,
                &mut target.stream,
                rounds,
            );
            self.schedule_child_layer_sample(
                child_layer_id,
                child_layer,
                &mut target.stream,
                rounds,
            );
            segment_start = cmd_idx + 1;
        }

        if self.command_segment_has_draws(cmds, segment_start, command_count) {
            let target = self.layer_command_target(layer_id, texture_index, bbox, &mut target)?;
            self.push_layer_batches(
                layer_idx,
                segment_start,
                command_count,
                &mut target.stream,
                rounds,
            );
        }

        let Some(mut target) = target else {
            return Ok(None);
        };
        let ready_round = self.finish_stream_segment(&mut target.stream, rounds);
        if let Some(filter) = target.filter {
            let allocation_filter = target
                .allocation
                .filter
                .expect("filter target must have scratch allocations");
            rounds.ensure_exists(ready_round, self.pools);
            rounds.rounds[ready_round].push_filter(
                target.region.texture.texture_index,
                &mut *self.buffers,
                FilterOp {
                    layer_region: target.region,
                    scratches: allocation_filter
                        .map(|scratch| scratch.map(|scratch| scratch.texture.region)),
                    filter_data_offset: filter.data_offset,
                    gpu_filter: filter.data,
                },
            );
        }
        Ok(Some(ScheduledLayer {
            sample: self.layer_sample(layer_id, target.region),
            allocation: target.allocation,
            region: target.region,
            round_idx: ready_round,
        }))
    }

    fn schedule_child_layer_sample(
        &mut self,
        layer_id: u32,
        layer: ScheduledLayer,
        state: &mut CommandStreamState,
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
            rounds.with_draw_builder(state, self.pools, &mut *self.buffers, |builder| {
                builder.push_layer_fill(
                    layer.sample,
                    props.opacity,
                    props.clip_path.as_ref(),
                    self.strip_storage,
                );
            });
            state.backdrop_bbox.union(layer.sample.bbox);
            state.sampled_layers.push(layer_id);
            self.finish_stream_segment(state, rounds);
            return;
        }

        let parent_ready_round = self.finish_stream_segment(state, rounds);
        let source_bbox = layer.sample.bbox;
        let blend_round = parent_ready_round.max(layer.round_idx);
        let affected_bbox = if blend_mode.is_destructive() {
            let mut bbox = state.backdrop_bbox;
            bbox.union(source_bbox);
            bbox
        } else {
            source_bbox
        };
        let bbox = affected_bbox.intersect(state.target.layer_region().scene_bbox);
        if bbox.is_empty() {
            self.consume_child_layer(layer_id, blend_round, rounds);
            state.round_idx = state.round_idx.max(blend_round);
            return;
        }

        rounds.ensure_exists(blend_round, self.pools);
        let parent_region = state.target.layer_region();
        rounds.rounds[blend_round].scratch_texture_clears[BLEND_SCRATCH_INDEX]
            .push(parent_region.blend_scratch_clear_rect(bbox));
        rounds.rounds[blend_round].push_blend(
            parent_region.texture.texture_index,
            &mut *self.buffers,
            BlendOp {
                parent_region,
                child_region: layer.sample.source,
                blend_bbox: bbox,
                blend_mode,
                opacity,
            },
        );
        self.consume_child_layer(layer_id, blend_round, rounds);
        state.backdrop_bbox = match blend_mode.compose {
            Compose::Clear => RectU16::INVERTED,
            Compose::Copy | Compose::SrcOut => source_bbox,
            Compose::SrcIn | Compose::DestIn => state.backdrop_bbox.intersect(source_bbox),
            _ => affected_bbox,
        };
        state.round_idx = blend_round + 1;
    }

    fn schedule_empty_destructive_blend(
        &mut self,
        state: &mut CommandStreamState,
        blend_mode: BlendMode,
        opacity: f32,
        rounds: &mut Rounds,
    ) {
        let parent_ready_round = self.finish_stream_segment(state, rounds);
        let affected_bbox = state.backdrop_bbox;
        let bbox = affected_bbox.intersect(state.target.layer_region().scene_bbox);
        if bbox.is_empty() {
            return;
        }

        rounds.ensure_exists(parent_ready_round, self.pools);
        let parent_region = state.target.layer_region();
        rounds.rounds[parent_ready_round].scratch_texture_clears[BLEND_SCRATCH_INDEX]
            .push(parent_region.blend_scratch_clear_rect(bbox));
        rounds.rounds[parent_ready_round].push_blend(
            parent_region.texture.texture_index,
            &mut *self.buffers,
            BlendOp {
                parent_region,
                child_region: LayerTextureRegion::empty_for_blend(bbox),
                blend_bbox: bbox,
                blend_mode,
                opacity,
            },
        );
        state.backdrop_bbox = match blend_mode.compose {
            Compose::Clear | Compose::Copy | Compose::SrcIn | Compose::SrcOut => RectU16::INVERTED,
            _ => affected_bbox,
        };
        state.round_idx = parent_ready_round + 1;
    }

    fn finish_stream_segment(
        &mut self,
        state: &mut CommandStreamState,
        rounds: &mut Rounds,
    ) -> usize {
        for layer_id in core::mem::take(&mut state.sampled_layers) {
            self.consume_child_layer(layer_id, state.round_idx, rounds);
        }

        state.round_idx
    }

    fn consume_child_layer(&mut self, layer_id: u32, round_idx: usize, rounds: &mut Rounds) {
        let Some(scheduled_layer) = self.layer_allocations[layer_id as usize].take() else {
            return;
        };

        rounds.ensure_exists(round_idx, self.pools);
        let clear_region = scheduled_layer.allocation.texture.clear_region();
        rounds.rounds[round_idx].layer_texture_clears[clear_region.texture_index]
            .push(clear_region.rect);
        if let Some(filter) = scheduled_layer.allocation.filter {
            for scratch in filter.into_iter().flatten() {
                let clear_region = scratch.texture.clear_region();
                rounds.rounds[round_idx].scratch_texture_clears[clear_region.texture_index]
                    .push(clear_region.rect);
            }
        }
        self.cursor
            .release_after(scheduled_layer.allocation, round_idx);
    }

    fn allocate_region(
        &mut self,
        texture_index: usize,
        bbox: RectU16,
        scratch_count: usize,
    ) -> Result<(Allocation<LayerAllocation>, LayerTextureRegion), RenderError> {
        let request = LayerAllocationRequest::new(
            texture_index,
            (bbox.width(), bbox.height()),
            scratch_count,
        );
        if !request.fits_texture(self.layer_texture_size) {
            return Err(RenderError::AtlasError(AtlasError::NoSpaceAvailable));
        }

        let Some(allocation) = self.cursor.allocate(request) else {
            return Err(RenderError::AtlasError(AtlasError::NoSpaceAvailable));
        };

        let region = LayerTextureRegion {
            texture: allocation.allocation.texture.region,
            scene_bbox: bbox,
        };

        Ok((allocation, region))
    }

    fn layer_sample(&self, layer_id: u32, allocation: LayerTextureRegion) -> LayerSample {
        let layer = &self.scene.recorder.layers[layer_id as usize];
        let RecordedLayerKind::Filter { placement, .. } = &layer.kind else {
            return LayerSample {
                source: allocation,
                bbox: layer.bbox,
                source_origin: (0, 0),
            };
        };

        LayerSample {
            source: LayerTextureRegion {
                texture: TextureRegion {
                    texture_index: allocation.texture.texture_index,
                    rect: RectU16::new(
                        allocation.texture.rect.x0 + placement.src_x,
                        allocation.texture.rect.y0 + placement.src_y,
                        allocation.texture.rect.x0 + placement.src_x + placement.dest_bbox.width(),
                        allocation.texture.rect.y0 + placement.src_y + placement.dest_bbox.height(),
                    ),
                },
                scene_bbox: placement.dest_bbox,
            },
            bbox: placement.dest_bbox,
            source_origin: (0, 0),
        }
    }

    fn layer_command_target<'b>(
        &mut self,
        layer_id: u32,
        texture_index: usize,
        bbox: RectU16,
        target: &'b mut Option<LayerCommandTarget>,
    ) -> Result<&'b mut LayerCommandTarget, RenderError> {
        if target.is_none() {
            let filter = self.filter_context.get(layer_id);
            let (allocation, region) = self.allocate_region(
                texture_index,
                bbox,
                filter.map_or(0, PreparedGpuFilter::scratch_count),
            )?;
            let stream = CommandStreamState::new(
                RenderTarget::Layer(region),
                allocation.round_idx,
                None,
                region.scene_bbox,
            );
            *target = Some(LayerCommandTarget {
                allocation: allocation.allocation,
                region,
                filter,
                stream,
            });
        }

        Ok(target.as_mut().expect("layer target must be initialized"))
    }

    fn push_layer_batches(
        &mut self,
        layer_idx: usize,
        start: usize,
        end: usize,
        state: &mut CommandStreamState,
        rounds: &mut Rounds,
    ) {
        self.push_command_batches(
            &self.scene.recorder.layers[layer_idx].cmds,
            start,
            end,
            state,
            rounds,
        );
    }

    fn command_segment_has_draws(&self, cmds: &[RecordedCmd], start: usize, end: usize) -> bool {
        cmds[start..end]
            .iter()
            .any(|cmd| matches!(cmd, RecordedCmd::Draws(_)))
    }
}

/// A layer that has been scheduled and can be sampled by its parent.
#[derive(Debug, Clone, Copy)]
struct ScheduledLayer {
    allocation: LayerAllocation,
    region: LayerTextureRegion,
    sample: LayerSample,
    round_idx: usize,
}

#[derive(Debug)]
struct LayerCommandTarget {
    allocation: LayerAllocation,
    region: LayerTextureRegion,
    filter: Option<PreparedGpuFilter>,
    stream: CommandStreamState,
}

#[derive(Debug)]
pub(super) struct CommandStreamState {
    pub(super) target: RenderTarget,
    pub(super) opaque: OpaqueStrips,
    pub(super) depth: DepthCounter,
    sampled_layers: Vec<u32>,
    backdrop_bbox: RectU16,
    pub(super) draw_bounds: RectU16,
    round_idx: usize,
}

impl CommandStreamState {
    fn new(
        target: RenderTarget,
        round_idx: usize,
        opaque: OpaqueStrips,
        draw_bounds: RectU16,
    ) -> Self {
        Self {
            target,
            opaque,
            depth: DepthCounter::default(),
            sampled_layers: Vec::new(),
            backdrop_bbox: RectU16::INVERTED,
            draw_bounds,
            round_idx,
        }
    }
}

impl Rounds {
    fn with_draw_builder(
        &mut self,
        state: &mut CommandStreamState,
        pools: &mut Pools,
        buffers: &mut ScheduleBuffers,
        f: impl FnOnce(&mut DrawBuilder<'_>),
    ) {
        self.ensure_exists(state.round_idx, pools);

        let target_draw = match state.target {
            RenderTarget::Root => self.rounds[state.round_idx].root_draw_mut(),
            RenderTarget::Layer(region) => {
                self.rounds[state.round_idx].layer_draw_mut(region.texture.texture_index)
            }
        };

        let mut builder = DrawBuilder::new(target_draw, buffers, state);
        f(&mut builder);
    }
}
