// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Round construction for the new hybrid scheduler.

use super::draw::{Draw, DrawBuilder, LayerSample};
use super::round::{BlendOp, FilterOp, Round, Rounds};
use super::timeline::{ResourceAllocator, Timeline};
use super::{LayerTextureRegion, LoadOp, RenderTarget, RootRenderTarget, ScratchRegion};
use crate::filter::{FILTER_ATLAS_PADDING, GpuFilterData};
use crate::scene::RecordedDraw;
use crate::{RenderError, Scene};
use alloc::vec::Vec;
use vello_common::encode::EncodedPaint;
use vello_common::filter::PreparedFilter;
use vello_common::geometry::RectU16;
use vello_common::multi_atlas::{AllocId, Atlas, AtlasError, AtlasId};
use vello_common::peniko::{BlendMode, Compose};
use vello_common::record::{Drawable, RecordedCmd, RecordedLayerKind};
use vello_common::strip_generator::StripStorage;
use vello_common::util::RectExt;

/// Builds concrete, executable rounds from a recorded scene.
#[derive(Debug)]
pub(super) struct ScheduleBuilder<'a> {
    scene: &'a Scene,
    strip_storage: &'a StripStorage,
    root_output_target: RootRenderTarget,
    paint_idxs: &'a [u32],
    encoded_paints: &'a [EncodedPaint],
    timeline: Timeline<LayerAtlasResource>,
    layer_allocations: Vec<Option<ScheduledLayer>>,
    filter_data_offsets: Vec<Option<u32>>,
    layer_texture_size: (u32, u32),
}

impl<'a> ScheduleBuilder<'a> {
    pub(super) fn new(
        scene: &'a Scene,
        strip_storage: &'a StripStorage,
        root_output_target: RootRenderTarget,
        paint_idxs: &'a [u32],
        encoded_paints: &'a [EncodedPaint],
        layer_texture_size: (u32, u32),
    ) -> Self {
        let layer_count = scene.recorder.layers.len();
        let mut filter_data_offsets = alloc::vec![None; layer_count];
        let mut filter_data_offset = 0;
        for layer_id in &scene.recorder.filter_layers {
            filter_data_offsets[*layer_id as usize] = Some(filter_data_offset);
            filter_data_offset += GpuFilterData::SIZE_TEXELS;
        }

        Self {
            scene,
            strip_storage,
            root_output_target,
            paint_idxs,
            encoded_paints,
            timeline: Timeline::new(LayerAtlasResource::new(layer_texture_size)),
            layer_allocations: alloc::vec![None; layer_count],
            filter_data_offsets,
            layer_texture_size,
        }
    }

    pub(super) fn build(&mut self) -> Result<Rounds, RenderError> {
        // TODO: Reuse round storage across frames so large schedules do not repeatedly allocate.
        let mut rounds = Rounds {
            rounds: alloc::vec![Round::default()],
        };

        // Walk the command tree left-to-right. Layer subtrees are scheduled lazily when their
        // parent command stream first needs to sample them. Atlas allocation is monotonic: pressure
        // advances the base round and applies completed cleanups instead of patching old states.
        self.schedule_root(&mut rounds)?;

        Ok(rounds)
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
        let texture_index = layer.depth & 1;
        if let Some(layer) =
            self.schedule_layer_command_stream(layer_id, texture_index, allocation_bbox, rounds)?
        {
            self.layer_allocations[layer_id as usize] = Some(layer);
        }

        Ok(())
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

        for (cmd_idx, cmd) in cmds.iter().enumerate().take(command_count) {
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
                    self.ensure_layer_command_target(layer_id, texture_index, bbox, &mut target)?;
                    let target = target.as_mut().unwrap();
                    self.push_layer_batches(layer_idx, segment_start, cmd_idx, &mut target.stream);
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

            self.ensure_layer_command_target(layer_id, texture_index, bbox, &mut target)?;
            let target = target.as_mut().unwrap();
            self.push_layer_batches(layer_idx, segment_start, cmd_idx, &mut target.stream);
            self.schedule_child_layer_sample(
                child_layer_id,
                child_layer,
                &mut target.stream,
                rounds,
            );
            segment_start = cmd_idx + 1;
        }

        if layer_segment_has_batches(cmds, segment_start, command_count) {
            self.ensure_layer_command_target(layer_id, texture_index, bbox, &mut target)?;
            let target = target.as_mut().unwrap();
            self.push_layer_batches(layer_idx, segment_start, command_count, &mut target.stream);
        }

        let Some(mut target) = target else {
            return Ok(None);
        };
        let ready_round = self.flush_stream_segment(&mut target.stream, rounds);
        if let Some(filter) = target.allocation.filter {
            self.ensure_round_exists(ready_round, rounds);
            rounds.rounds[ready_round].push_filter(FilterOp {
                layer_region: target.allocation.region,
                scratches: filter
                    .scratches
                    .map(|scratch| scratch.map(|scratch| scratch.region)),
                filter_data_offset: filter.filter_data_offset,
                gpu_filter: filter.gpu_filter,
            });
        }
        Ok(Some(ScheduledLayer {
            sample: self.layer_sample(layer_id, target.allocation.region),
            allocation: target.allocation,
            round_idx: ready_round,
        }))
    }

    fn ensure_layer_command_target(
        &mut self,
        layer_id: u32,
        texture_index: usize,
        bbox: RectU16,
        target: &mut Option<LayerCommandTarget>,
    ) -> Result<(), RenderError> {
        if target.is_some() {
            return Ok(());
        }

        let allocation = self.allocate_region(
            texture_index,
            bbox,
            self.filter_allocation_request(layer_id),
        )?;
        let stream = CommandStreamState::new(
            RenderTarget::Layer(allocation.region),
            allocation.round_idx,
            LoadOp::Load,
            allocation.region.scene_bbox,
        );
        *target = Some(LayerCommandTarget { allocation, stream });

        Ok(())
    }

    fn push_layer_batches(
        &self,
        layer_idx: usize,
        start: usize,
        end: usize,
        state: &mut CommandStreamState,
    ) {
        for cmd in &self.scene.recorder.layers[layer_idx].cmds[start..end] {
            let RecordedCmd::Draws(range) = cmd else {
                continue;
            };

            for draw in &self.scene.recorder.draws[range.start as usize..range.end as usize] {
                state
                    .backdrop_bbox
                    .union(recorded_draw_bbox(draw, self.strip_storage));
                state.builder.push_draw(
                    draw,
                    self.strip_storage,
                    self.encoded_paints,
                    self.paint_idxs,
                );
            }
        }
    }

    fn schedule_root(&mut self, rounds: &mut Rounds) -> Result<(), RenderError> {
        let cmds = &self.scene.recorder.root_cmds;
        if self.scene.recorder.root_is_blend_target {
            let bbox = RectU16::new(0, 0, self.scene.width, self.scene.height);
            if bbox.is_empty() {
                return Ok(());
            }

            let allocation = self.allocate_region(0, bbox, None)?;
            let target = RenderTarget::Layer(allocation.region);
            let ready_round = self.schedule_command_stream_with_load(
                cmds,
                target,
                allocation.round_idx,
                LoadOp::Clear,
                rounds,
            )?;

            self.ensure_round_exists(ready_round, rounds);
            let mut draw = DrawBuilder::new(
                RenderTarget::Root(self.root_output_target).allows_opaque_pass(),
                (0, 0),
                RectU16::new(0, 0, self.scene.width, self.scene.height),
            );
            draw.push_layer_ref(
                LayerSample {
                    source: allocation.region,
                    bbox: allocation.region.scene_bbox,
                    source_origin: (0, 0),
                },
                1.0,
                None,
                self.strip_storage,
            );
            let draw = draw.take_draw();
            if !draw.is_empty() {
                let final_target = match self.root_output_target {
                    RootRenderTarget::UserSurface => RootRenderTarget::UserSurfaceFromLayer0,
                    RootRenderTarget::AtlasLayer => RootRenderTarget::AtlasLayerFromLayer0,
                    other => other,
                };
                rounds.rounds[ready_round].push_pass(RenderTarget::Root(final_target), draw);
            }
            rounds.rounds[ready_round]
                .layer_clears
                .push(allocation.region);
            self.release_allocation_after_round(allocation, ready_round, rounds);
        } else {
            let target = RenderTarget::Root(self.root_output_target);
            let ready_round =
                self.schedule_command_stream(cmds, target, self.timeline.base_round(), rounds)?;
            self.ensure_round_exists(ready_round, rounds);
        }

        Ok(())
    }

    fn schedule_command_stream(
        &mut self,
        cmds: &[RecordedCmd],
        target: RenderTarget,
        start_round: usize,
        rounds: &mut Rounds,
    ) -> Result<usize, RenderError> {
        self.schedule_command_stream_with_load(cmds, target, start_round, LoadOp::Load, rounds)
    }

    fn schedule_command_stream_with_load(
        &mut self,
        cmds: &[RecordedCmd],
        target: RenderTarget,
        start_round: usize,
        initial_load_op: LoadOp,
        rounds: &mut Rounds,
    ) -> Result<usize, RenderError> {
        let mut state = CommandStreamState::new(
            target,
            start_round,
            initial_load_op,
            target_draw_bounds(target, self.scene),
        );

        for cmd in cmds {
            match cmd {
                RecordedCmd::Draws(range) => {
                    for draw in &self.scene.recorder.draws[range.start as usize..range.end as usize]
                    {
                        state
                            .backdrop_bbox
                            .union(recorded_draw_bbox(draw, self.strip_storage));
                        state.builder.push_draw(
                            draw,
                            self.strip_storage,
                            self.encoded_paints,
                            self.paint_idxs,
                        );
                    }
                }
                RecordedCmd::Layer(layer_id) => {
                    let layer_idx = *layer_id as usize;
                    self.schedule_layer_subtree(*layer_id, rounds)?;
                    let Some(layer) = self.layer_allocations[layer_idx] else {
                        let props = &self.scene.recorder.layers[layer_idx].props;
                        if props.blend_mode.is_destructive() {
                            self.schedule_empty_destructive_blend(
                                &mut state,
                                props.blend_mode,
                                props.opacity,
                                rounds,
                            );
                        }
                        continue;
                    };

                    self.schedule_child_layer_sample(*layer_id, layer, &mut state, rounds);
                }
            }
        }

        Ok(self.flush_stream_segment(&mut state, rounds))
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
            state.target.texture_index() == Some(layer.allocation.region.texture_index);
        if blend_mode == BlendMode::default() && !same_texture_as_target {
            state.round_idx = state.round_idx.max(required_round_for_layer_sample(
                state.target.texture_index(),
                layer.allocation.region.texture_index,
                layer.round_idx,
            ));
            state.builder.push_layer_ref(
                layer.sample,
                props.opacity,
                props.clip_path.as_ref(),
                self.strip_storage,
            );
            state.backdrop_bbox.union(layer.sample.bbox);
            state.sampled_layers.push(layer_id);
            self.flush_stream_segment(state, rounds);
            return;
        }

        let parent_ready_round = self.flush_stream_segment(state, rounds);
        let source_bbox = layer.sample.bbox;
        let blend_round = parent_ready_round.max(layer.round_idx);
        let bbox = blend_affected_bbox(state.backdrop_bbox, source_bbox, blend_mode.compose)
            .intersect(state.target.layer_region().scene_bbox);
        if bbox.is_empty() {
            self.consume_child_layer(layer_id, blend_round, rounds);
            state.round_idx = state.round_idx.max(blend_round);
            return;
        }

        self.ensure_round_exists(blend_round, rounds);
        rounds.rounds[blend_round].push_blend(BlendOp {
            parent_region: state.target.layer_region(),
            child_region: layer.sample.source,
            blend_bbox: bbox,
            blend_mode,
            opacity,
        });
        self.consume_child_layer(layer_id, blend_round, rounds);
        state.backdrop_bbox =
            blend_result_bbox(state.backdrop_bbox, source_bbox, blend_mode.compose);
        state.round_idx = blend_round + 1;
    }

    fn schedule_empty_destructive_blend(
        &mut self,
        state: &mut CommandStreamState,
        blend_mode: BlendMode,
        opacity: f32,
        rounds: &mut Rounds,
    ) {
        let parent_ready_round = self.flush_stream_segment(state, rounds);
        let bbox = blend_affected_bbox(state.backdrop_bbox, RectU16::INVERTED, blend_mode.compose)
            .intersect(state.target.layer_region().scene_bbox);
        if bbox.is_empty() {
            return;
        }

        self.ensure_round_exists(parent_ready_round, rounds);
        rounds.rounds[parent_ready_round].push_blend(BlendOp {
            parent_region: state.target.layer_region(),
            child_region: empty_child_region_for_blend(bbox),
            blend_bbox: bbox,
            blend_mode,
            opacity,
        });
        state.backdrop_bbox =
            blend_result_bbox(state.backdrop_bbox, RectU16::INVERTED, blend_mode.compose);
        state.round_idx = parent_ready_round + 1;
    }

    fn flush_stream_segment(
        &mut self,
        state: &mut CommandStreamState,
        rounds: &mut Rounds,
    ) -> usize {
        let draw = state.take_draw();
        if !draw.is_empty() {
            self.ensure_round_exists(state.round_idx, rounds);
            rounds.rounds[state.round_idx].push_pass_with_load(
                state.target,
                draw,
                state.take_load_op(),
            );
        }

        for layer_id in core::mem::take(&mut state.sampled_layers) {
            self.consume_child_layer(layer_id, state.round_idx, rounds);
        }

        state.round_idx
    }

    fn consume_child_layer(&mut self, layer_id: u32, round_idx: usize, rounds: &mut Rounds) {
        let Some(scheduled_layer) = self.layer_allocations[layer_id as usize].take() else {
            return;
        };

        self.ensure_round_exists(round_idx, rounds);
        rounds.rounds[round_idx]
            .layer_clears
            .push(scheduled_layer.allocation.clear_region);
        if let Some(filter) = scheduled_layer.allocation.filter {
            for scratch in filter.scratches.into_iter().flatten() {
                rounds.rounds[round_idx]
                    .scratch_clears
                    .push(scratch.clear_region);
            }
        }
        self.release_allocation_after_round(scheduled_layer.allocation, round_idx, rounds);
    }

    fn allocate_region(
        &mut self,
        texture_index: usize,
        bbox: RectU16,
        filter: Option<FilterAllocationRequest>,
    ) -> Result<LayerAllocation, RenderError> {
        let width = bbox.width();
        let height = bbox.height();
        let padding = if filter.is_some() {
            FILTER_ATLAS_PADDING
        } else {
            0
        };
        let allocation_width = u32::from(width).saturating_add(u32::from(padding) * 2);
        let allocation_height = u32::from(height).saturating_add(u32::from(padding) * 2);
        if allocation_width > self.layer_texture_size.0
            || allocation_height > self.layer_texture_size.1
        {
            return Err(RenderError::AtlasError(AtlasError::NoSpaceAvailable));
        }

        let request = LayerAllocationRequest {
            texture_index,
            bbox,
            width,
            height,
            padding,
            allocation_width,
            allocation_height,
            filter,
        };
        let Some(scheduled) = self.timeline.allocate(request) else {
            return Err(RenderError::AtlasError(AtlasError::NoSpaceAvailable));
        };

        let mut allocation = scheduled.allocation;
        allocation.round_idx = scheduled.round_idx;

        Ok(allocation)
    }

    fn filter_allocation_request(&self, layer_id: u32) -> Option<FilterAllocationRequest> {
        let layer = &self.scene.recorder.layers[layer_id as usize];
        let RecordedLayerKind::Filter { filter_data, .. } = &layer.kind else {
            return None;
        };

        let prepared_filter = PreparedFilter::new(&filter_data.filter, &filter_data.transform);
        let gpu_filter = GpuFilterData::from(&prepared_filter);
        let is_multi_pass = gpu_filter.is_multi_pass();
        Some(FilterAllocationRequest {
            scratch_count: if is_multi_pass { 2 } else { 1 },
            filter_data_offset: self.filter_data_offsets[layer_id as usize]
                .expect("filter layer must have a filter data offset"),
            gpu_filter,
        })
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
                x: allocation.x + placement.src_x,
                y: allocation.y + placement.src_y,
                scene_bbox: placement.dest_bbox,
                width: placement.dest_bbox.width(),
                height: placement.dest_bbox.height(),
                ..allocation
            },
            bbox: placement.dest_bbox,
            source_origin: (0, 0),
        }
    }

    fn ensure_round_exists(&mut self, round_idx: usize, rounds: &mut Rounds) {
        ensure_round_exists(rounds, round_idx);
    }

    fn release_allocation_after_round(
        &mut self,
        allocation: LayerAllocation,
        round_idx: usize,
        rounds: &mut Rounds,
    ) {
        ensure_round_exists(rounds, round_idx);
        self.timeline.release_after(allocation, round_idx);
    }
}

fn ensure_round_exists(rounds: &mut Rounds, round_idx: usize) {
    while rounds.rounds.len() <= round_idx {
        rounds.rounds.push(Round::default());
    }
}

#[derive(Debug)]
struct LayerAtlasResource {
    atlases: [Atlas; 2],
    scratch_atlases: [Atlas; 2],
}

impl LayerAtlasResource {
    fn new(layer_texture_size: (u32, u32)) -> Self {
        Self {
            atlases: [
                Atlas::new(AtlasId::new(0), layer_texture_size.0, layer_texture_size.1),
                Atlas::new(AtlasId::new(1), layer_texture_size.0, layer_texture_size.1),
            ],
            scratch_atlases: [
                Atlas::new(AtlasId::new(0), layer_texture_size.0, layer_texture_size.1),
                Atlas::new(AtlasId::new(1), layer_texture_size.0, layer_texture_size.1),
            ],
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LayerAllocationRequest {
    texture_index: usize,
    bbox: RectU16,
    width: u16,
    height: u16,
    padding: u16,
    allocation_width: u32,
    allocation_height: u32,
    filter: Option<FilterAllocationRequest>,
}

#[derive(Debug, Clone, Copy)]
struct FilterAllocationRequest {
    scratch_count: usize,
    filter_data_offset: u32,
    gpu_filter: GpuFilterData,
}

impl ResourceAllocator for LayerAtlasResource {
    type Request = LayerAllocationRequest;
    type Allocation = LayerAllocation;

    fn allocate(&mut self, request: Self::Request) -> Option<Self::Allocation> {
        let padding = u32::from(request.padding);
        let allocation = self.atlases[request.texture_index]
            .allocate(request.allocation_width, request.allocation_height)?;
        let mut scratch_allocations: [Option<ScratchAllocation>; 2] = [None, None];

        if let Some(filter) = request.filter {
            for (scratch_index, scratch) in scratch_allocations
                .iter_mut()
                .enumerate()
                .take(filter.scratch_count)
            {
                let Some(allocation) = self.scratch_atlases[scratch_index]
                    .allocate(request.allocation_width, request.allocation_height)
                else {
                    for (allocated_index, allocated_scratch) in
                        scratch_allocations.iter().enumerate()
                    {
                        if let Some(allocated_scratch) = allocated_scratch {
                            self.scratch_atlases[allocated_index].deallocate(
                                allocated_scratch.alloc_id,
                                allocated_scratch.allocation_width,
                                allocated_scratch.allocation_height,
                            );
                        }
                    }
                    self.atlases[request.texture_index].deallocate(
                        allocation.id,
                        request.allocation_width,
                        request.allocation_height,
                    );
                    return None;
                };
                *scratch = Some(ScratchAllocation {
                    region: ScratchRegion {
                        texture_index: scratch_index,
                        rect: RectU16::new(
                            atlas_coord(allocation.x + padding),
                            atlas_coord(allocation.y + padding),
                            atlas_coord(allocation.x + padding + u32::from(request.width)),
                            atlas_coord(allocation.y + padding + u32::from(request.height)),
                        ),
                    },
                    clear_region: ScratchRegion {
                        texture_index: scratch_index,
                        rect: RectU16::new(
                            atlas_coord(allocation.x),
                            atlas_coord(allocation.y),
                            atlas_coord(allocation.x + request.allocation_width),
                            atlas_coord(allocation.y + request.allocation_height),
                        ),
                    },
                    alloc_id: allocation.id,
                    allocation_width: request.allocation_width,
                    allocation_height: request.allocation_height,
                });
            }
        }

        Some(LayerAllocation {
            region: LayerTextureRegion {
                texture_index: request.texture_index,
                x: atlas_coord(allocation.x + padding),
                y: atlas_coord(allocation.y + padding),
                width: request.width,
                height: request.height,
                scene_bbox: request.bbox,
            },
            clear_region: LayerTextureRegion {
                texture_index: request.texture_index,
                x: atlas_coord(allocation.x),
                y: atlas_coord(allocation.y),
                width: atlas_coord(request.allocation_width),
                height: atlas_coord(request.allocation_height),
                scene_bbox: request.bbox,
            },
            filter: request.filter.map(|filter| FilterAllocation {
                scratches: scratch_allocations,
                filter_data_offset: filter.filter_data_offset,
                gpu_filter: filter.gpu_filter,
            }),
            round_idx: 0,
            alloc_id: allocation.id,
            allocation_width: request.allocation_width,
            allocation_height: request.allocation_height,
        })
    }

    fn release(&mut self, allocation: Self::Allocation) {
        self.atlases[allocation.region.texture_index].deallocate(
            allocation.alloc_id,
            allocation.allocation_width,
            allocation.allocation_height,
        );
        if let Some(filter) = allocation.filter {
            for scratch in filter.scratches.into_iter().flatten() {
                self.scratch_atlases[scratch.region.texture_index].deallocate(
                    scratch.alloc_id,
                    scratch.allocation_width,
                    scratch.allocation_height,
                );
            }
        }
    }
}

/// A layer that has been scheduled and can be sampled by its parent.
#[derive(Debug, Clone, Copy)]
struct ScheduledLayer {
    allocation: LayerAllocation,
    sample: LayerSample,
    round_idx: usize,
}

/// A layer texture region plus the allocator handle needed to release it.
#[derive(Debug, Clone, Copy)]
struct LayerAllocation {
    region: LayerTextureRegion,
    clear_region: LayerTextureRegion,
    filter: Option<FilterAllocation>,
    round_idx: usize,
    alloc_id: AllocId,
    allocation_width: u32,
    allocation_height: u32,
}

#[derive(Debug, Clone, Copy)]
struct FilterAllocation {
    scratches: [Option<ScratchAllocation>; 2],
    filter_data_offset: u32,
    gpu_filter: GpuFilterData,
}

#[derive(Debug, Clone, Copy)]
struct ScratchAllocation {
    region: ScratchRegion,
    clear_region: ScratchRegion,
    alloc_id: AllocId,
    allocation_width: u32,
    allocation_height: u32,
}

#[derive(Debug)]
struct LayerCommandTarget {
    allocation: LayerAllocation,
    stream: CommandStreamState,
}

#[derive(Debug)]
struct CommandStreamState {
    target: RenderTarget,
    builder: DrawBuilder,
    sampled_layers: Vec<u32>,
    backdrop_bbox: RectU16,
    next_load_op: LoadOp,
    round_idx: usize,
}

impl CommandStreamState {
    fn new(
        target: RenderTarget,
        round_idx: usize,
        initial_load_op: LoadOp,
        draw_bounds: RectU16,
    ) -> Self {
        Self {
            target,
            builder: DrawBuilder::new(
                target.allows_opaque_pass(),
                target.geometry_offset(),
                draw_bounds,
            ),
            sampled_layers: Vec::new(),
            backdrop_bbox: RectU16::INVERTED,
            next_load_op: initial_load_op,
            round_idx,
        }
    }

    fn take_load_op(&mut self) -> LoadOp {
        let load_op = self.next_load_op;
        self.next_load_op = LoadOp::Load;
        load_op
    }

    fn take_draw(&mut self) -> Draw {
        self.builder.take_draw()
    }
}

fn target_draw_bounds(target: RenderTarget, scene: &Scene) -> RectU16 {
    match target {
        RenderTarget::Root(_) => RectU16::new(0, 0, scene.width, scene.height),
        RenderTarget::Layer(region) => region.scene_bbox,
    }
}

fn can_sample_layer_in_same_round(parent_texture_index: usize, child_texture_index: usize) -> bool {
    parent_texture_index != child_texture_index
        && layer_texture_order(child_texture_index) < layer_texture_order(parent_texture_index)
}

fn required_round_for_layer_sample(
    parent_texture_index: Option<usize>,
    child_texture_index: usize,
    child_round: usize,
) -> usize {
    match parent_texture_index {
        Some(parent_texture_index)
            if can_sample_layer_in_same_round(parent_texture_index, child_texture_index) =>
        {
            child_round
        }
        Some(_) => child_round + 1,
        None => child_round,
    }
}

fn layer_texture_order(texture_index: usize) -> usize {
    match texture_index {
        1 => 0,
        0 => 1,
        _ => texture_index,
    }
}

fn blend_affected_bbox(backdrop_bbox: RectU16, source_bbox: RectU16, compose: Compose) -> RectU16 {
    match compose {
        Compose::Clear
        | Compose::Copy
        | Compose::SrcIn
        | Compose::SrcOut
        | Compose::DestIn
        | Compose::DestAtop => union_bbox(backdrop_bbox, source_bbox),
        _ => source_bbox,
    }
}

fn blend_result_bbox(backdrop_bbox: RectU16, source_bbox: RectU16, compose: Compose) -> RectU16 {
    match compose {
        Compose::Clear => RectU16::INVERTED,
        Compose::Copy => source_bbox,
        Compose::SrcIn | Compose::DestIn => backdrop_bbox.intersect(source_bbox),
        Compose::SrcOut => source_bbox,
        Compose::Dest => backdrop_bbox,
        Compose::DestOut => backdrop_bbox,
        Compose::DestAtop => union_bbox(backdrop_bbox, source_bbox),
        _ => union_bbox(backdrop_bbox, source_bbox),
    }
}

fn union_bbox(mut a: RectU16, b: RectU16) -> RectU16 {
    a.union(b);
    a
}

fn empty_child_region_for_blend(bbox: RectU16) -> LayerTextureRegion {
    LayerTextureRegion {
        texture_index: 0,
        x: 0,
        y: 0,
        width: 0,
        height: 0,
        scene_bbox: RectU16::new(bbox.x0, bbox.y0, bbox.x0, bbox.y0),
    }
}

fn atlas_coord(value: u32) -> u16 {
    u16::try_from(value).expect("atlas coordinate must fit into u16")
}

fn layer_segment_has_batches(cmds: &[RecordedCmd], start: usize, end: usize) -> bool {
    cmds[start..end]
        .iter()
        .any(|cmd| matches!(cmd, RecordedCmd::Draws(_)))
}

fn recorded_draw_bbox(draw: &RecordedDraw, strip_storage: &StripStorage) -> RectU16 {
    let strips = match draw {
        RecordedDraw::Path(path) => &strip_storage.strips[path.strips.clone()],
        RecordedDraw::Rect(_) => &[],
    };
    draw.bbox(strips)
}
