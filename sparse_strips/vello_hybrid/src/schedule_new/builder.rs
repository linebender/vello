// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Schedule construction for the new hybrid scheduler.

use super::round::{BlendOp, FilterOp, Round, Schedule};
use super::timeline::{ResourceAllocator, Timeline};
use super::{
    Draw, FilterScratchRegion, LayerTextureRegion, LoadOp, RenderTarget, RootRenderTarget,
};
use crate::filter::{FILTER_ATLAS_PADDING, GpuFilterData};
use crate::scene::{FastPathRect, RecordedDraw};
use crate::schedule::{GpuStripBuilder, RectPart, make_gpu_rect, process_paint, split_rect};
use crate::{RenderError, Scene};
use alloc::vec::Vec;
use vello_common::encode::EncodedPaint;
use vello_common::filter::PreparedFilter;
use vello_common::geometry::RectU16;
use vello_common::multi_atlas::{AllocId, Atlas, AtlasError, AtlasId};
use vello_common::paint::Paint;
use vello_common::peniko::{BlendMode, Compose};
use vello_common::record::{Drawable, LayerClip, RecordedCmd, RecordedLayerKind};
use vello_common::strip_generator::StripStorage;
use vello_common::tile::Tile;

const COLOR_SOURCE_LAYER: u32 = 1;

/// Builds a concrete, executable schedule from a recorded scene.
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

    pub(super) fn build(&mut self) -> Result<Schedule, RenderError> {
        self.validate_layers()?;

        let mut schedule = Schedule {
            rounds: alloc::vec![Round::default()],
        };

        // Walk the command tree left-to-right. Layer subtrees are scheduled lazily when their
        // parent command stream first needs to sample them. Atlas allocation is monotonic: pressure
        // advances the base round and applies completed cleanups instead of patching old states.
        self.schedule_root(&mut schedule)?;

        Ok(schedule)
    }

    fn validate_layers(&self) -> Result<(), RenderError> {
        for layer in &self.scene.recorder.layers {
            ensure_supported_layer(layer)?;
        }

        Ok(())
    }

    fn schedule_layer_subtree(
        &mut self,
        layer_id: u32,
        schedule: &mut Schedule,
    ) -> Result<(), RenderError> {
        if self.layer_allocations[layer_id as usize].is_some() {
            return Ok(());
        }

        let layer = &self.scene.recorder.layers[layer_id as usize];
        let bbox = layer.bbox;
        if bbox.is_empty() {
            return Ok(());
        }

        let texture_index = layer.depth & 1;
        if let Some(layer) =
            self.schedule_layer_command_stream(layer_id, texture_index, bbox, schedule)?
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
        schedule: &mut Schedule,
    ) -> Result<Option<ScheduledLayer>, RenderError> {
        let layer_idx = layer_id as usize;
        let cmds = &self.scene.recorder.layers[layer_idx].cmds;
        let command_count = cmds.len();
        let mut target = None;
        let mut segment_start = 0;

        for cmd_idx in 0..command_count {
            let child_layer_id = match &cmds[cmd_idx] {
                RecordedCmd::Draws(_) => continue,
                RecordedCmd::Layer(child_layer_id) => *child_layer_id,
            };

            let child_layer_idx = child_layer_id as usize;
            self.schedule_layer_subtree(child_layer_id, schedule)?;
            let Some(child_layer) = self.layer_allocations[child_layer_idx] else {
                let props = &self.scene.recorder.layers[child_layer_idx].props;
                let blend_mode = props.blend_mode;
                let opacity = props.opacity;
                if blend_mode.is_destructive() {
                    self.ensure_layer_command_target(
                        layer_id,
                        texture_index,
                        bbox,
                        schedule,
                        &mut target,
                    )?;
                    let target = target.as_mut().unwrap();
                    self.push_layer_batches(layer_idx, segment_start, cmd_idx, &mut target.stream);
                    self.schedule_empty_destructive_blend(
                        &mut target.stream,
                        blend_mode,
                        opacity,
                        schedule,
                    );
                    segment_start = cmd_idx + 1;
                }
                continue;
            };

            self.ensure_layer_command_target(layer_id, texture_index, bbox, schedule, &mut target)?;
            let target = target.as_mut().unwrap();
            self.push_layer_batches(layer_idx, segment_start, cmd_idx, &mut target.stream);
            self.schedule_child_layer_sample(
                child_layer_id,
                child_layer,
                &mut target.stream,
                schedule,
            );
            segment_start = cmd_idx + 1;
        }

        if layer_segment_has_batches(cmds, segment_start, command_count) {
            self.ensure_layer_command_target(layer_id, texture_index, bbox, schedule, &mut target)?;
            let target = target.as_mut().unwrap();
            self.push_layer_batches(layer_idx, segment_start, command_count, &mut target.stream);
        }

        let Some(mut target) = target else {
            return Ok(None);
        };
        let ready_round = self.flush_stream_segment(&mut target.stream, schedule);
        if let Some(filter) = target.allocation.filter {
            self.ensure_schedule_round_exists(ready_round, schedule);
            schedule.rounds[ready_round].push_filter(FilterOp {
                layer: target.allocation.region,
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
        schedule: &mut Schedule,
        target: &mut Option<LayerCommandTarget>,
    ) -> Result<(), RenderError> {
        if target.is_some() {
            return Ok(());
        }

        let allocation = self.allocate_region(
            texture_index,
            bbox,
            self.filter_allocation_request(layer_id),
            self.timeline.base_round(),
            schedule,
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
                state.backdrop_bbox.union(recorded_draw_bbox(
                    draw,
                    self.strip_storage,
                    self.scene.width,
                ));
                state.builder.push_draw(
                    draw,
                    self.strip_storage,
                    self.encoded_paints,
                    self.paint_idxs,
                );
            }
        }
    }

    fn schedule_root(&mut self, schedule: &mut Schedule) -> Result<(), RenderError> {
        let cmds = &self.scene.recorder.root_cmds;
        if self.scene.recorder.root_is_blend_target {
            let bbox = RectU16::new(0, 0, self.scene.width, self.scene.height);
            if bbox.is_empty() {
                return Ok(());
            }

            let allocation = self.allocate_region(
                0,
                bbox,
                None,
                self.timeline.base_round(),
                schedule,
            )?;
            let target = RenderTarget::Layer(allocation.region);
            let ready_round = self.schedule_command_stream_with_load(
                cmds,
                target,
                allocation.round_idx,
                LoadOp::Clear,
                schedule,
            )?;

            self.ensure_schedule_round_exists(ready_round, schedule);
            let mut draw = DrawBuilder::new(
                RenderTarget::Root(self.root_output_target).allows_opaque_pass(),
                (0, 0),
                RectU16::new(0, 0, self.scene.width, self.scene.height),
            );
            draw.push_layer_ref(
                LayerSample {
                    region: allocation.region,
                    source_origin: (0, 0),
                },
                1.0,
                None,
                self.strip_storage,
            );
            let draw = draw.finish();
            if !draw.is_empty() {
                let final_target = match self.root_output_target {
                    RootRenderTarget::UserSurface => RootRenderTarget::UserSurfaceFromLayer0,
                    RootRenderTarget::AtlasLayer => RootRenderTarget::AtlasLayerFromLayer0,
                    other => other,
                };
                schedule.rounds[ready_round].push_pass(RenderTarget::Root(final_target), draw);
            }
            schedule.rounds[ready_round]
                .clear_layer_regions
                .push(allocation.region);
            self.release_allocation_after_round(allocation, ready_round, schedule);
        } else {
            let target = RenderTarget::Root(self.root_output_target);
            let ready_round =
                self.schedule_command_stream(cmds, target, self.timeline.base_round(), schedule)?;
            self.ensure_schedule_round_exists(ready_round, schedule);
        }

        Ok(())
    }

    fn schedule_command_stream(
        &mut self,
        cmds: &[RecordedCmd],
        target: RenderTarget,
        start_round: usize,
        schedule: &mut Schedule,
    ) -> Result<usize, RenderError> {
        self.schedule_command_stream_with_load(cmds, target, start_round, LoadOp::Load, schedule)
    }

    fn schedule_command_stream_with_load(
        &mut self,
        cmds: &[RecordedCmd],
        target: RenderTarget,
        start_round: usize,
        initial_load_op: LoadOp,
        schedule: &mut Schedule,
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
                        state.backdrop_bbox.union(recorded_draw_bbox(
                            draw,
                            self.strip_storage,
                            self.scene.width,
                        ));
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
                    self.schedule_layer_subtree(*layer_id, schedule)?;
                    let Some(layer) = self.layer_allocations[layer_idx] else {
                        let props = &self.scene.recorder.layers[layer_idx].props;
                        if props.blend_mode.is_destructive() {
                            self.schedule_empty_destructive_blend(
                                &mut state,
                                props.blend_mode,
                                props.opacity,
                                schedule,
                            );
                        }
                        continue;
                    };

                    self.schedule_child_layer_sample(*layer_id, layer, &mut state, schedule);
                }
            }
        }

        Ok(self.flush_stream_segment(&mut state, schedule))
    }

    fn schedule_child_layer_sample(
        &mut self,
        layer_id: u32,
        layer: ScheduledLayer,
        state: &mut CommandStreamState,
        schedule: &mut Schedule,
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
            state.backdrop_bbox.union(layer.sample.region.scene_bbox);
            state.sampled_layers.push(layer_id);
            self.flush_stream_segment(state, schedule);
            return;
        }

        let parent_ready_round = self.flush_stream_segment(state, schedule);
        let source_bbox = layer.sample.region.scene_bbox;
        let blend_round = parent_ready_round.max(layer.round_idx);
        let bbox = blend_affected_bbox(state.backdrop_bbox, source_bbox, blend_mode.compose)
            .intersect(state.target.layer_region().scene_bbox);
        if bbox.is_empty() {
            self.consume_child_layer(layer_id, blend_round, schedule);
            state.round_idx = state.round_idx.max(blend_round);
            return;
        }

        self.ensure_schedule_round_exists(blend_round, schedule);
        schedule.rounds[blend_round].push_blend(BlendOp {
            parent: state.target.layer_region(),
            source: layer.sample.region,
            bbox,
            blend_mode,
            opacity,
        });
        self.consume_child_layer(layer_id, blend_round, schedule);
        state.backdrop_bbox =
            blend_result_bbox(state.backdrop_bbox, source_bbox, blend_mode.compose);
        state.round_idx = blend_round + 1;
    }

    fn schedule_empty_destructive_blend(
        &mut self,
        state: &mut CommandStreamState,
        blend_mode: BlendMode,
        opacity: f32,
        schedule: &mut Schedule,
    ) {
        let parent_ready_round = self.flush_stream_segment(state, schedule);
        let bbox = blend_affected_bbox(state.backdrop_bbox, RectU16::INVERTED, blend_mode.compose)
            .intersect(state.target.layer_region().scene_bbox);
        if bbox.is_empty() {
            return;
        }

        self.ensure_schedule_round_exists(parent_ready_round, schedule);
        schedule.rounds[parent_ready_round].push_blend(BlendOp {
            parent: state.target.layer_region(),
            source: empty_source_region_for_blend(bbox),
            bbox,
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
        schedule: &mut Schedule,
    ) -> usize {
        let draw = state.take_draw();
        if !draw.is_empty() {
            self.ensure_schedule_round_exists(state.round_idx, schedule);
            schedule.rounds[state.round_idx].push_pass_with_load(
                state.target,
                draw,
                state.take_load_op(),
            );
        }

        for layer_id in core::mem::take(&mut state.sampled_layers) {
            self.consume_child_layer(layer_id, state.round_idx, schedule);
        }

        state.round_idx
    }

    fn consume_child_layer(&mut self, layer_id: u32, round_idx: usize, schedule: &mut Schedule) {
        let Some(scheduled_layer) = self.layer_allocations[layer_id as usize].take() else {
            return;
        };

        self.ensure_schedule_round_exists(round_idx, schedule);
        schedule.rounds[round_idx]
            .clear_layer_regions
            .push(scheduled_layer.allocation.clear_region);
        if let Some(filter) = scheduled_layer.allocation.filter {
            for scratch in filter.scratches.into_iter().flatten() {
                schedule.rounds[round_idx]
                    .clear_filter_scratch_regions
                    .push(scratch.clear_region);
            }
        }
        self.release_allocation_after_round(scheduled_layer.allocation, round_idx, schedule);
    }

    fn allocate_region(
        &mut self,
        texture_index: usize,
        bbox: RectU16,
        filter: Option<FilterAllocationRequest>,
        earliest_round: usize,
        schedule: &mut Schedule,
    ) -> Result<LayerAllocation, RenderError> {
        let width = u32::from(bbox.width());
        let height = u32::from(bbox.height());
        let padding = filter.map_or(0, |filter| u32::from(filter.padding));
        let allocation_width = width.saturating_add(padding * 2);
        let allocation_height = height.saturating_add(padding * 2);
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
            allocation_width,
            allocation_height,
            filter,
        };
        let Some(scheduled) = self
            .timeline
            .allocate_after(request, earliest_round, |round_idx| {
                ensure_schedule_round_exists(schedule, round_idx);
            })
        else {
            return Err(RenderError::AtlasError(AtlasError::NoSpaceAvailable));
        };

        let mut allocation = scheduled.allocation;
        allocation.round_idx = scheduled.round_idx;

        if allocation.has_padding() {
            self.ensure_schedule_round_exists(allocation.round_idx, schedule);
            let round = &mut schedule.rounds[allocation.round_idx];
            round.prepare_layer_regions.push(allocation.clear_region);
            if let Some(filter) = allocation.filter {
                for scratch in filter.scratches.into_iter().flatten() {
                    round
                        .prepare_filter_scratch_regions
                        .push(scratch.clear_region);
                }
            }
        }

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
            padding: if is_multi_pass {
                FILTER_ATLAS_PADDING
            } else {
                0
            },
            filter_data_offset: self.filter_data_offsets[layer_id as usize]
                .expect("filter layer must have a filter data offset"),
            gpu_filter,
        })
    }

    fn layer_sample(&self, layer_id: u32, allocation: LayerTextureRegion) -> LayerSample {
        let layer = &self.scene.recorder.layers[layer_id as usize];
        let RecordedLayerKind::Filter { placement, .. } = &layer.kind else {
            return LayerSample {
                region: allocation,
                source_origin: (0, 0),
            };
        };

        LayerSample {
            region: LayerTextureRegion {
                x: allocation.x + u32::from(placement.src_x),
                y: allocation.y + u32::from(placement.src_y),
                scene_bbox: placement.dest_bbox,
                width: u32::from(placement.dest_bbox.width()),
                height: u32::from(placement.dest_bbox.height()),
                ..allocation
            },
            source_origin: (0, 0),
        }
    }

    fn ensure_schedule_round_exists(&mut self, round_idx: usize, schedule: &mut Schedule) {
        ensure_schedule_round_exists(schedule, round_idx);
    }

    fn release_allocation_after_round(
        &mut self,
        allocation: LayerAllocation,
        round_idx: usize,
        schedule: &mut Schedule,
    ) {
        self.timeline
            .release_after(allocation, round_idx, |round_idx| {
                ensure_schedule_round_exists(schedule, round_idx);
            });
    }
}

fn ensure_schedule_round_exists(schedule: &mut Schedule, round_idx: usize) {
    while schedule.rounds.len() <= round_idx {
        schedule.rounds.push(Round::default());
    }
}

#[derive(Debug)]
struct LayerAtlasResource {
    atlases: [Atlas; 2],
    filter_scratch_atlases: [Atlas; 2],
}

impl LayerAtlasResource {
    fn new(layer_texture_size: (u32, u32)) -> Self {
        Self {
            atlases: [
                Atlas::new(AtlasId::new(0), layer_texture_size.0, layer_texture_size.1),
                Atlas::new(AtlasId::new(1), layer_texture_size.0, layer_texture_size.1),
            ],
            filter_scratch_atlases: [
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
    width: u32,
    height: u32,
    allocation_width: u32,
    allocation_height: u32,
    filter: Option<FilterAllocationRequest>,
}

#[derive(Debug, Clone, Copy)]
struct FilterAllocationRequest {
    scratch_count: usize,
    padding: u16,
    filter_data_offset: u32,
    gpu_filter: GpuFilterData,
}

impl ResourceAllocator for LayerAtlasResource {
    type Request = LayerAllocationRequest;
    type Allocation = LayerAllocation;

    fn allocate(&mut self, request: Self::Request) -> Option<Self::Allocation> {
        let padding = request.filter.map_or(0, |filter| u32::from(filter.padding));
        let allocation = self.atlases[request.texture_index]
            .allocate(request.allocation_width, request.allocation_height)?;
        let mut scratch_allocations: [Option<FilterScratchAllocation>; 2] = [None, None];

        if let Some(filter) = request.filter {
            for (scratch_index, scratch) in scratch_allocations
                .iter_mut()
                .enumerate()
                .take(filter.scratch_count)
            {
                let Some(allocation) = self.filter_scratch_atlases[scratch_index]
                    .allocate(request.allocation_width, request.allocation_height)
                else {
                    for (allocated_index, allocated_scratch) in
                        scratch_allocations.iter().enumerate()
                    {
                        if let Some(allocated_scratch) = allocated_scratch {
                            self.filter_scratch_atlases[allocated_index].deallocate(
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
                *scratch = Some(FilterScratchAllocation {
                    region: FilterScratchRegion {
                        texture_index: scratch_index,
                        x: allocation.x + padding,
                        y: allocation.y + padding,
                        width: request.width,
                        height: request.height,
                    },
                    clear_region: FilterScratchRegion {
                        texture_index: scratch_index,
                        x: allocation.x,
                        y: allocation.y,
                        width: request.allocation_width,
                        height: request.allocation_height,
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
                x: allocation.x + padding,
                y: allocation.y + padding,
                width: request.width,
                height: request.height,
                scene_bbox: request.bbox,
            },
            clear_region: LayerTextureRegion {
                texture_index: request.texture_index,
                x: allocation.x,
                y: allocation.y,
                width: request.allocation_width,
                height: request.allocation_height,
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
                self.filter_scratch_atlases[scratch.region.texture_index].deallocate(
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

impl LayerAllocation {
    fn has_padding(self) -> bool {
        self.allocation_width != self.region.width || self.allocation_height != self.region.height
    }
}

#[derive(Debug, Clone, Copy)]
struct FilterAllocation {
    scratches: [Option<FilterScratchAllocation>; 2],
    filter_data_offset: u32,
    gpu_filter: GpuFilterData,
}

#[derive(Debug, Clone, Copy)]
struct FilterScratchAllocation {
    region: FilterScratchRegion,
    clear_region: FilterScratchRegion,
    alloc_id: AllocId,
    allocation_width: u32,
    allocation_height: u32,
}

#[derive(Debug, Clone, Copy)]
struct LayerSample {
    region: LayerTextureRegion,
    source_origin: (u16, u16),
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
        let replacement = DrawBuilder {
            draw: Draw::default(),
            depth: self.builder.depth,
            allow_opaque_pass: self.builder.allow_opaque_pass,
            geometry_offset: self.builder.geometry_offset,
            draw_bounds: self.builder.draw_bounds,
        };
        let builder = core::mem::replace(&mut self.builder, replacement);
        builder.finish()
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

fn empty_source_region_for_blend(bbox: RectU16) -> LayerTextureRegion {
    LayerTextureRegion {
        texture_index: 0,
        x: 0,
        y: 0,
        width: 0,
        height: 0,
        scene_bbox: RectU16::new(bbox.x0, bbox.y0, bbox.x0, bbox.y0),
    }
}

fn layer_segment_has_batches(cmds: &[RecordedCmd], start: usize, end: usize) -> bool {
    cmds[start..end]
        .iter()
        .any(|cmd| matches!(cmd, RecordedCmd::Draws(_)))
}

fn recorded_draw_bbox(
    draw: &RecordedDraw,
    strip_storage: &StripStorage,
    viewport_width: u16,
) -> RectU16 {
    let strips = match draw {
        RecordedDraw::Path(path) => &strip_storage.strips[path.strips.clone()],
        RecordedDraw::Rect(_) => &[],
    };
    draw.bbox(strips, viewport_width)
}

fn ensure_supported_layer(layer: &vello_common::record::RecordedLayer) -> Result<(), RenderError> {
    if layer.props.mask.is_some() {
        return Err(RenderError::UnsupportedFeature(
            "mask layers are not supported by schedule_new yet",
        ));
    }
    if layer
        .props
        .clip_path
        .as_ref()
        .is_some_and(|clip_path| clip_path.thread_idx != 0)
    {
        return Err(RenderError::UnsupportedFeature(
            "multi-threaded clip layers are not supported by schedule_new yet",
        ));
    }
    Ok(())
}

#[derive(Debug)]
struct DrawBuilder {
    draw: Draw,
    depth: DepthCounter,
    allow_opaque_pass: bool,
    geometry_offset: (i32, i32),
    draw_bounds: RectU16,
}

impl DrawBuilder {
    fn new(allow_opaque_pass: bool, geometry_offset: (i32, i32), draw_bounds: RectU16) -> Self {
        Self {
            draw: Draw::default(),
            depth: DepthCounter::default(),
            allow_opaque_pass,
            geometry_offset,
            draw_bounds,
        }
    }

    fn push_draw(
        &mut self,
        draw: &RecordedDraw,
        strip_storage: &StripStorage,
        encoded_paints: &[EncodedPaint],
        paint_idxs: &[u32],
    ) {
        match draw {
            RecordedDraw::Path(path) => {
                self.push_path(
                    path.strips.clone(),
                    path.paint.clone(),
                    strip_storage,
                    encoded_paints,
                    paint_idxs,
                );
            }
            RecordedDraw::Rect(rect) => {
                self.push_rect(&rect.rect, encoded_paints, paint_idxs);
            }
        }
    }

    fn push_path(
        &mut self,
        strips: core::ops::Range<usize>,
        paint: Paint,
        strip_storage: &StripStorage,
        encoded_paints: &[EncodedPaint],
        paint_idxs: &[u32],
    ) {
        let strips = &strip_storage.strips[strips];

        if strips.is_empty() {
            return;
        }

        let is_opaque = self.allow_opaque_pass && is_paint_opaque(&paint, encoded_paints);
        let depth_index = self.depth.next(is_opaque);

        for i in 0..strips.len() - 1 {
            let strip = &strips[i];
            let y = strip.y;

            if strip.x >= self.draw_bounds.x1 || y < self.draw_bounds.y0 || y >= self.draw_bounds.y1
            {
                continue;
            }

            let next_strip = &strips[i + 1];
            let col = strip.alpha_idx() / u32::from(Tile::HEIGHT);
            let next_col = next_strip.alpha_idx() / u32::from(Tile::HEIGHT);
            let strip_width = next_col.saturating_sub(col) as u16;
            let target_y = offset_coord(y, self.geometry_offset.1);

            if strip_width > 0 {
                let strip_x0 = strip.x;
                let strip_x1 = strip_x0
                    .saturating_add(strip_width)
                    .min(self.draw_bounds.x1);
                let x0 = strip_x0.max(self.draw_bounds.x0);
                let x1 = strip_x1.min(self.draw_bounds.x1);
                if x1 > x0 {
                    let width = x1 - x0;
                    let col_offset = col + u32::from(x0 - strip_x0);
                    let target_x0 = offset_coord(x0, self.geometry_offset.0);
                    let processed = process_paint(&paint, encoded_paints, (x0, y), paint_idxs);
                    self.draw.push_alpha(
                        GpuStripBuilder::at_surface(target_x0, target_y, width)
                            .with_sparse(width, col_offset)
                            .paint(processed.payload, processed.paint, depth_index),
                        processed.external_texture_id,
                    );
                }
            }

            if next_strip.fill_gap() && strip.strip_y() == next_strip.strip_y() {
                let gap_x0 = strip.x.saturating_add(strip_width);
                let gap_x1 = next_strip.x.min(self.draw_bounds.x1);
                let x0 = gap_x0.max(self.draw_bounds.x0);
                let x1 = gap_x1.min(self.draw_bounds.x1);
                if x1 > x0 {
                    let target_x0 = offset_coord(x0, self.geometry_offset.0);
                    let processed = process_paint(&paint, encoded_paints, (x0, y), paint_idxs);
                    let strip = GpuStripBuilder::at_surface(target_x0, target_y, x1 - x0).paint(
                        processed.payload,
                        processed.paint,
                        depth_index,
                    );
                    if is_opaque {
                        self.draw.push_opaque(strip);
                    } else {
                        self.draw.push_alpha(strip, processed.external_texture_id);
                    }
                }
            }
        }
    }

    fn push_rect(
        &mut self,
        rect: &FastPathRect,
        encoded_paints: &[EncodedPaint],
        paint_idxs: &[u32],
    ) {
        let Some(rect) = clipped_fast_rect(rect, self.draw_bounds) else {
            return;
        };
        let is_opaque = self.allow_opaque_pass && is_paint_opaque(&rect.paint, encoded_paints);
        let depth_index = self.depth.next(is_opaque);
        pack_rectangle_into_gpu(
            &rect,
            encoded_paints,
            paint_idxs,
            depth_index,
            is_opaque,
            self.geometry_offset,
            &mut self.draw,
        );
    }

    fn push_layer_ref(
        &mut self,
        sample: LayerSample,
        opacity: f32,
        clip_path: Option<&LayerClip>,
        strip_storage: &StripStorage,
    ) {
        // TODO: Add optimization to not emit strips outside of clip bbox.
        if let Some(clip_path) = clip_path {
            self.push_clipped_layer_ref(sample, opacity, clip_path, strip_storage);
            return;
        }

        let depth_index = self.depth.next(false);
        let bbox = sample.region.scene_bbox.intersect(self.draw_bounds);
        if bbox.is_empty() {
            return;
        }

        // Layer samples are encoded as image-like rect paints. Geometry is transformed into the
        // target allocation, while the payload points at the source atlas coordinate.
        self.draw.push_alpha(
            make_gpu_rect(
                offset_rect_part(
                    RectPart {
                        x: bbox.x0,
                        y: bbox.y0,
                        width: bbox.width(),
                        height: bbox.height(),
                        frac: 0,
                    },
                    self.geometry_offset,
                ),
                layer_sample_payload(sample, bbox.x0, bbox.y0),
                layer_paint(opacity),
                depth_index,
            ),
            None,
        );
    }

    // TODO: Deduplicate this with vello cpu.
    fn push_clipped_layer_ref(
        &mut self,
        sample: LayerSample,
        opacity: f32,
        clip_path: &LayerClip,
        strip_storage: &StripStorage,
    ) {
        let strips = &strip_storage.strips[clip_path.strip_range.clone()];
        let sample_bbox = sample.region.scene_bbox.intersect(self.draw_bounds);
        if strips.len() < 2 || clip_path.bbox.is_empty() || sample_bbox.is_empty() {
            return;
        }

        let depth_index = self.depth.next(false);
        let paint = layer_paint(opacity);

        for i in 0..strips.len() - 1 {
            let strip = &strips[i];
            let next_strip = &strips[i + 1];

            if strip.is_sentinel() {
                continue;
            }

            let y = strip.y;
            if y < sample_bbox.y0 || y >= sample_bbox.y1 {
                continue;
            }

            let strip_width = strip.width_to(next_strip);
            if strip_width > 0 {
                let x0 = strip.x.max(sample_bbox.x0);
                let x1 = strip.x.saturating_add(strip_width).min(sample_bbox.x1);
                if x1 > x0 {
                    let col = strip.alpha_idx() / u32::from(Tile::HEIGHT);
                    let col_offset = col + u32::from(x0 - strip.x);
                    self.draw.push_alpha(
                        GpuStripBuilder::at_surface(
                            offset_coord(x0, self.geometry_offset.0),
                            offset_coord(y, self.geometry_offset.1),
                            x1 - x0,
                        )
                        .with_sparse(x1 - x0, col_offset)
                        .paint(
                            layer_sample_payload(sample, x0, y),
                            paint,
                            depth_index,
                        ),
                        None,
                    );
                }
            }

            if next_strip.fill_gap() && next_strip.y == strip.y {
                let x0 = strip.x.saturating_add(strip_width).max(sample_bbox.x0);
                let x1 = if next_strip.is_sentinel() {
                    sample_bbox.x1
                } else {
                    next_strip.x.min(sample_bbox.x1)
                };
                if x1 > x0 {
                    self.draw.push_alpha(
                        GpuStripBuilder::at_surface(
                            offset_coord(x0, self.geometry_offset.0),
                            offset_coord(y, self.geometry_offset.1),
                            x1 - x0,
                        )
                        .paint(
                            layer_sample_payload(sample, x0, y),
                            paint,
                            depth_index,
                        ),
                        None,
                    );
                }
            }
        }
    }

    fn finish(mut self) -> Draw {
        self.draw.opaque.reverse();
        self.draw
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct DepthCounter {
    count: u32,
}

impl DepthCounter {
    #[inline(always)]
    fn next(&mut self, opaque: bool) -> u32 {
        self.count += opaque as u32;
        self.count
    }
}

fn is_paint_opaque(paint: &Paint, encoded_paints: &[EncodedPaint]) -> bool {
    match paint {
        Paint::Solid(color) => color.is_opaque(),
        Paint::Indexed(indexed_paint) => match encoded_paints.get(indexed_paint.index()) {
            Some(EncodedPaint::Image(image)) => {
                !image.may_have_transparency
                    && image.sampler.alpha == 1.0
                    && image.tint.is_none_or(|t| t.color.components[3] >= 1.0)
            }
            Some(EncodedPaint::ExternalTexture(_)) => false,
            Some(EncodedPaint::Gradient(gradient)) => !gradient.may_have_transparency,
            Some(EncodedPaint::BlurredRoundedRect(_)) => false,
            None => unreachable!("Paint must be in encoded paints"),
        },
    }
}

fn pack_rectangle_into_gpu(
    rect: &FastPathRect,
    encoded_paints: &[EncodedPaint],
    paint_idxs: &[u32],
    depth_index: u32,
    is_opaque: bool,
    geometry_offset: (i32, i32),
    draw: &mut Draw,
) {
    let split = split_rect(rect);

    let mut is_first = true;
    for part in [
        Some(split.main),
        split.top,
        split.bottom,
        split.left,
        split.right,
    ]
    .into_iter()
    .flatten()
    {
        let processed = process_paint(&rect.paint, encoded_paints, (part.x, part.y), paint_idxs);
        let strip = make_gpu_rect(
            offset_rect_part(part, geometry_offset),
            processed.payload,
            processed.paint,
            depth_index,
        );
        if is_first && is_opaque && part.frac == 0 {
            draw.push_opaque(strip);
        } else {
            draw.push_alpha(strip, processed.external_texture_id);
        }
        is_first = false;
    }
}

fn clipped_fast_rect(rect: &FastPathRect, bbox: RectU16) -> Option<FastPathRect> {
    let x0 = rect.x0.max(f32::from(bbox.x0));
    let y0 = rect.y0.max(f32::from(bbox.y0));
    let x1 = rect.x1.min(f32::from(bbox.x1));
    let y1 = rect.y1.min(f32::from(bbox.y1));

    (x0 < x1 && y0 < y1).then(|| FastPathRect {
        x0,
        y0,
        x1,
        y1,
        paint: rect.paint.clone(),
    })
}

fn offset_rect_part(part: RectPart, offset: (i32, i32)) -> RectPart {
    RectPart {
        x: offset_coord(part.x, offset.0),
        y: offset_coord(part.y, offset.1),
        ..part
    }
}

fn offset_coord(coord: u16, offset: i32) -> u16 {
    let coord = i32::from(coord) + offset;
    debug_assert!((0..=i32::from(u16::MAX)).contains(&coord));
    coord as u16
}

fn pack_u16_pair(x: u32, y: u32) -> u32 {
    debug_assert!(x <= u32::from(u16::MAX));
    debug_assert!(y <= u32::from(u16::MAX));
    (x & 0xffff) | ((y & 0xffff) << 16)
}

fn layer_sample_payload(sample: LayerSample, x: u16, y: u16) -> u32 {
    let source = sample.region;
    let source_x =
        source.x + u32::from(sample.source_origin.0) + u32::from(x - source.scene_bbox.x0);
    let source_y =
        source.y + u32::from(sample.source_origin.1) + u32::from(y - source.scene_bbox.y0);
    pack_u16_pair(source_x, source_y)
}

fn layer_paint(opacity: f32) -> u32 {
    (COLOR_SOURCE_LAYER << 29) | u32::from(opacity_to_u8(opacity))
}

fn opacity_to_u8(opacity: f32) -> u8 {
    (opacity.clamp(0.0, 1.0) * 255.0).round() as u8
}
