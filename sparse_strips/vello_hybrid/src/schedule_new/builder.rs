// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Schedule construction for the new hybrid scheduler.

use super::round::{
    AllocationRetryDebug, BlendOp, LayerScheduleDebug, Round, Schedule, ScheduleDebugStats,
};
use super::{Draw, LayerTextureRegion, LoadOp, RenderTarget, RootRenderTarget};
use crate::scene::{FastPathRect, RecordedDraw};
use crate::schedule::{
    GpuStripBuilder, RectPart, Scheduler as ExistingScheduler, make_gpu_rect, split_rect,
};
use crate::{RenderError, Scene};
use alloc::vec::Vec;
use vello_common::encode::EncodedPaint;
use vello_common::geometry::RectU16;
use vello_common::multi_atlas::{AllocId, Atlas, AtlasError, AtlasId};
use vello_common::paint::Paint;
use vello_common::peniko::{BlendMode, Compose};
use vello_common::record::{Drawable, LayerClip, RecordedCmd, RecordedLayerKind};
use vello_common::strip_generator::StripStorage;
use vello_common::tile::Tile;
use vello_common::util::RectExt;

const COLOR_SOURCE_LAYER: u32 = 1;

/// Builds a concrete, executable schedule from a recorded scene.
#[derive(Debug)]
pub(super) struct ScheduleBuilder<'a> {
    scene: &'a Scene,
    strip_storage: &'a StripStorage,
    root_output_target: RootRenderTarget,
    paint_idxs: &'a [u32],
    encoded_paints: &'a [EncodedPaint],
    round_states: Vec<RoundState>,
    layer_allocations: Vec<Option<ScheduledLayer>>,
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
        let initial_state = RoundState::new(layer_texture_size);
        Self {
            scene,
            strip_storage,
            root_output_target,
            paint_idxs,
            encoded_paints,
            round_states: alloc::vec![initial_state],
            layer_allocations: alloc::vec![None; layer_count],
            layer_texture_size,
        }
    }

    pub(super) fn build(&mut self) -> Result<Schedule, RenderError> {
        self.validate_layers()?;

        let mut schedule = Schedule {
            rounds: alloc::vec![Round::default()],
            debug: self.initial_debug_stats(),
        };

        // Walk the command tree left-to-right. Each encountered layer schedules its own subtree
        // first. Command streams are then materialized in order, flushing draw segments around
        // blend invocations so backdrop/source dependencies stay explicit.
        self.schedule_children(&self.scene.recorder.root_cmds, &mut schedule)?;
        self.schedule_root(&mut schedule)?;

        Ok(schedule)
    }

    fn validate_layers(&self) -> Result<(), RenderError> {
        for layer in &self.scene.recorder.layers {
            ensure_plain_layer(layer)?;
        }

        Ok(())
    }

    fn schedule_children(
        &mut self,
        cmds: &[RecordedCmd],
        schedule: &mut Schedule,
    ) -> Result<(), RenderError> {
        for layer_id in child_layer_ids(cmds) {
            self.schedule_layer_subtree(layer_id, schedule)?;
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

        let layer_idx = layer_id as usize;
        let cmds = &self.scene.recorder.layers[layer_idx].cmds;
        // This is an ordered DFS post-order walk: all child layers in this stream are attempted
        // before the containing layer, but we never jump ahead into later sibling subtrees.
        self.schedule_children(cmds, schedule)?;

        let layer = &self.scene.recorder.layers[layer_idx];
        let bbox = layer.props.clip_path.as_ref().map_or(layer.bbox, |clip| {
            let mut bbox = layer.bbox;
            bbox.union(clip.bbox.snap_to_tile_coordinates());
            bbox
        });
        if bbox.is_empty() {
            return Ok(());
        }

        let texture_index = layer.depth & 1;
        let layer_depth = layer.depth;
        let has_clip = layer.props.clip_path.is_some();
        let has_default_blend = layer.props.blend_mode == BlendMode::default();
        let is_destructive_blend = layer.props.blend_mode.is_destructive();
        let opacity = layer.props.opacity;
        let command_count = cmds.len();
        let child_layer_count = child_layer_ids(cmds).count();
        let batch_count = cmds
            .iter()
            .filter(|cmd| matches!(cmd, RecordedCmd::Batch(_)))
            .count();
        let target_round = 0;
        let allocation =
            self.allocate_layer(layer_id, texture_index, bbox, target_round, schedule)?;
        let target = RenderTarget::Layer(allocation.allocation.region);
        let ready_round =
            self.schedule_command_stream(cmds, target, allocation.round_idx, schedule)?;
        self.layer_allocations[layer_id as usize] = Some(ScheduledLayer {
            allocation: allocation.allocation,
            round_idx: ready_round,
        });
        schedule.debug.scheduled_layers.push(LayerScheduleDebug {
            layer_id,
            depth: layer_depth,
            texture_index,
            command_count,
            child_layer_count,
            batch_count,
            allocated_round: allocation.round_idx,
            ready_round,
            bbox,
            has_clip,
            has_default_blend,
            is_destructive_blend,
            opacity,
        });

        Ok(())
    }

    fn initial_debug_stats(&self) -> ScheduleDebugStats {
        let mut depth_counts = Vec::<(usize, usize)>::new();
        for layer in &self.scene.recorder.layers {
            if let Some((_, count)) = depth_counts
                .iter_mut()
                .find(|(depth, _)| *depth == layer.depth)
            {
                *count += 1;
            } else {
                depth_counts.push((layer.depth, 1));
            }
        }
        depth_counts.sort_by_key(|(depth, _)| *depth);

        ScheduleDebugStats {
            layer_count: self.scene.recorder.layers.len(),
            root_cmd_count: self.scene.recorder.root_cmds.len(),
            root_child_layer_count: child_layer_ids(&self.scene.recorder.root_cmds).count(),
            root_is_blend_target: self.scene.recorder.root_is_blend_target,
            layer_texture_size: self.layer_texture_size,
            depth_counts,
            ..ScheduleDebugStats::default()
        }
    }

    fn schedule_root(&mut self, schedule: &mut Schedule) -> Result<(), RenderError> {
        let cmds = &self.scene.recorder.root_cmds;
        if self.scene.recorder.root_is_blend_target {
            let bbox = RectU16::new(0, 0, self.scene.width, self.scene.height);
            if bbox.is_empty() {
                return Ok(());
            }

            let allocation = self.allocate_region(0, bbox, 0, schedule)?;
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
            );
            draw.push_layer_ref(allocation.region, 1.0, None, self.strip_storage);
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
            let ready_round = self.schedule_command_stream(cmds, target, 0, schedule)?;
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
        let mut state = CommandStreamState::new(target, start_round, initial_load_op);

        for cmd in cmds {
            match cmd {
                RecordedCmd::Batch(range) => {
                    for draw in &self.scene.recorder.draws[range.start as usize..range.end as usize]
                    {
                        state.backdrop_bbox.union(draw.bbox());
                        state.builder.push_draw(
                            draw,
                            self.strip_storage,
                            self.scene,
                            self.encoded_paints,
                            self.paint_idxs,
                        );
                    }
                }
                RecordedCmd::Layer(layer_id) => {
                    let layer_idx = *layer_id as usize;
                    let props = &self.scene.recorder.layers[layer_idx].props;
                    let Some(layer) = self.layer_allocations[layer_idx] else {
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

                    let same_texture_as_target =
                        state.target.texture_index() == Some(layer.allocation.region.texture_index);
                    if props.blend_mode == BlendMode::default() && !same_texture_as_target {
                        state.round_idx = state.round_idx.max(required_round_for_layer_sample(
                            state.target.texture_index(),
                            layer.allocation.region.texture_index,
                            layer.round_idx,
                        ));
                        state.builder.push_layer_ref(
                            layer.allocation.region,
                            props.opacity,
                            props.clip_path.as_ref(),
                            self.strip_storage,
                        );
                        state
                            .backdrop_bbox
                            .union(layer.allocation.region.scene_bbox);
                        state.sampled_layers.push(*layer_id);
                        continue;
                    }

                    let parent_ready_round = self.flush_stream_segment(&mut state, schedule);
                    let source_bbox = layer.allocation.region.scene_bbox;
                    let bbox = blend_affected_bbox(
                        state.backdrop_bbox,
                        source_bbox,
                        props.blend_mode.compose,
                    )
                    .intersect(state.target.layer_region().scene_bbox);
                    if !bbox.is_empty() {
                        let blend_round = parent_ready_round.max(layer.round_idx);
                        self.ensure_schedule_round_exists(blend_round, schedule);
                        schedule.rounds[blend_round].push_blend(BlendOp {
                            parent: state.target.layer_region(),
                            source: layer.allocation.region,
                            bbox,
                            blend_mode: props.blend_mode,
                            opacity: props.opacity,
                        });
                        self.consume_child_layer(*layer_id, blend_round, schedule);
                        state.backdrop_bbox = blend_result_bbox(
                            state.backdrop_bbox,
                            source_bbox,
                            props.blend_mode.compose,
                        );
                        state.round_idx = blend_round + 1;
                    }
                }
            }
        }

        Ok(self.flush_stream_segment(&mut state, schedule))
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
        if draw.is_empty() {
            return state.round_idx;
        }

        self.ensure_schedule_round_exists(state.round_idx, schedule);
        schedule.rounds[state.round_idx].push_pass_with_load(
            state.target,
            draw,
            state.take_load_op(),
        );

        for layer_id in core::mem::take(&mut state.sampled_layers) {
            self.consume_child_layer(layer_id, state.round_idx, schedule);
        }

        state.round_idx
    }

    fn consume_child_layer(&mut self, layer_id: u32, round_idx: usize, schedule: &mut Schedule) {
        let Some(scheduled_layer) = self.layer_allocations[layer_id as usize].take() else {
            return;
        };

        schedule.rounds[round_idx]
            .clear_layer_regions
            .push(scheduled_layer.allocation.region);
        self.release_allocation_after_round(scheduled_layer.allocation, round_idx, schedule);
    }

    fn allocate_layer(
        &mut self,
        layer_id: u32,
        texture_index: usize,
        bbox: RectU16,
        earliest_round: usize,
        schedule: &mut Schedule,
    ) -> Result<ScheduledLayer, RenderError> {
        let allocation = self.allocate_region(texture_index, bbox, earliest_round, schedule)?;
        let scheduled_layer = ScheduledLayer {
            allocation,
            round_idx: allocation.round_idx,
        };
        self.layer_allocations[layer_id as usize] = Some(scheduled_layer);
        Ok(scheduled_layer)
    }

    fn allocate_region(
        &mut self,
        texture_index: usize,
        bbox: RectU16,
        earliest_round: usize,
        schedule: &mut Schedule,
    ) -> Result<LayerAllocation, RenderError> {
        let width = u32::from(bbox.width());
        let height = u32::from(bbox.height());
        if width > self.layer_texture_size.0 || height > self.layer_texture_size.1 {
            return Err(RenderError::AtlasError(AtlasError::NoSpaceAvailable));
        }

        let mut round_idx = earliest_round;
        let mut attempts = 0;
        loop {
            attempts += 1;
            self.ensure_round_state_exists(round_idx, schedule);

            if let Some(allocation) =
                self.round_states[round_idx].atlases[texture_index].allocate(width, height)
            {
                let layer_allocation = LayerAllocation {
                    region: LayerTextureRegion {
                        texture_index,
                        x: allocation.x,
                        y: allocation.y,
                        width,
                        height,
                        scene_bbox: bbox,
                    },
                    round_idx,
                    alloc_id: allocation.id,
                };
                if self.reserve_existing_future_rounds(layer_allocation, round_idx) {
                    schedule.debug.allocation_attempts += attempts;
                    schedule.debug.allocation_retries += attempts.saturating_sub(1);
                    if round_idx != earliest_round {
                        schedule
                            .debug
                            .allocation_retry_events
                            .push(AllocationRetryDebug {
                                texture_index,
                                earliest_round,
                                allocated_round: round_idx,
                                attempts,
                                bbox,
                            });
                    }
                    return Ok(layer_allocation);
                }

                self.round_states[round_idx].deallocate(layer_allocation);
                round_idx += 1;
                continue;
            }

            // If nothing is due to be freed after this round, another retry would see the same
            // allocator state and would spin forever.
            if self.round_states[round_idx]
                .pending_deallocations
                .is_empty()
            {
                return Err(RenderError::AtlasError(AtlasError::NoSpaceAvailable));
            }

            round_idx += 1;
        }
    }

    fn ensure_schedule_round_exists(&mut self, round_idx: usize, schedule: &mut Schedule) {
        while schedule.rounds.len() <= round_idx {
            schedule.rounds.push(Round::default());
        }
    }

    fn ensure_round_state_exists(&mut self, round_idx: usize, schedule: &mut Schedule) {
        self.ensure_schedule_round_exists(round_idx, schedule);
        while self.round_states.len() <= round_idx {
            let previous_round = self.round_states.len() - 1;
            let mut next_state = self.round_states[previous_round].clone_for_next_round();
            for allocation in &self.round_states[previous_round].pending_deallocations {
                next_state.deallocate(*allocation);
            }
            self.round_states.push(next_state);
        }
    }

    fn release_allocation_after_round(
        &mut self,
        allocation: LayerAllocation,
        round_idx: usize,
        schedule: &mut Schedule,
    ) {
        self.ensure_round_state_exists(round_idx, schedule);
        self.round_states[round_idx]
            .pending_deallocations
            .push(allocation);

        // Future round states inherit from this round. If they already exist, keep them consistent
        // with the newly scheduled cleanup.
        for state in &mut self.round_states[round_idx + 1..] {
            state.deallocate(allocation);
        }
    }

    fn reserve_existing_future_rounds(
        &mut self,
        allocation: LayerAllocation,
        round_idx: usize,
    ) -> bool {
        // Future states may already exist because some earlier subtree needed a later round. The
        // newly allocated layer is live until consumed, so reserve it in those existing states too.
        let mut reservations = Vec::new();
        for state_idx in round_idx + 1..self.round_states.len() {
            let state = &mut self.round_states[state_idx];
            let region = allocation.region;
            let Some(future_allocation) =
                state.atlases[region.texture_index].allocate(region.width, region.height)
            else {
                rollback_future_reservations(&mut self.round_states, &reservations);
                return false;
            };

            if future_allocation.x != region.x || future_allocation.y != region.y {
                state.atlases[region.texture_index].deallocate(
                    future_allocation.id,
                    region.width,
                    region.height,
                );
                rollback_future_reservations(&mut self.round_states, &reservations);
                return false;
            }

            state.alloc_id_aliases.push(AllocIdAlias {
                texture_index: region.texture_index,
                original: allocation.alloc_id,
                local: future_allocation.id,
            });
            reservations.push(FutureReservation {
                round_idx: state_idx,
                texture_index: region.texture_index,
                alloc_id: future_allocation.id,
                width: region.width,
                height: region.height,
                original_alloc_id: allocation.alloc_id,
            });
        }

        true
    }
}

fn rollback_future_reservations(
    round_states: &mut [RoundState],
    reservations: &[FutureReservation],
) {
    for reservation in reservations.iter().rev() {
        let state = &mut round_states[reservation.round_idx];
        state.atlases[reservation.texture_index].deallocate(
            reservation.alloc_id,
            reservation.width,
            reservation.height,
        );
        state.alloc_id_aliases.retain(|alias| {
            !(alias.texture_index == reservation.texture_index
                && alias.original == reservation.original_alloc_id)
        });
    }
}

#[derive(Debug, Clone, Copy)]
struct FutureReservation {
    round_idx: usize,
    texture_index: usize,
    alloc_id: AllocId,
    width: u32,
    height: u32,
    original_alloc_id: AllocId,
}

#[derive(Debug, Clone)]
struct RoundState {
    atlases: [Atlas; 2],
    pending_deallocations: Vec<LayerAllocation>,
    alloc_id_aliases: Vec<AllocIdAlias>,
}

#[derive(Debug, Clone, Copy)]
struct AllocIdAlias {
    texture_index: usize,
    original: AllocId,
    local: AllocId,
}

impl RoundState {
    fn new(layer_texture_size: (u32, u32)) -> Self {
        Self {
            atlases: [
                Atlas::new(AtlasId::new(0), layer_texture_size.0, layer_texture_size.1),
                Atlas::new(AtlasId::new(1), layer_texture_size.0, layer_texture_size.1),
            ],
            pending_deallocations: Vec::new(),
            alloc_id_aliases: Vec::new(),
        }
    }

    fn clone_for_next_round(&self) -> Self {
        Self {
            atlases: self.atlases.clone(),
            pending_deallocations: Vec::new(),
            alloc_id_aliases: self.alloc_id_aliases.clone(),
        }
    }

    fn deallocate(&mut self, allocation: LayerAllocation) {
        let alloc_id = self
            .alloc_id_aliases
            .iter()
            .find(|alias| {
                alias.texture_index == allocation.region.texture_index
                    && alias.original == allocation.alloc_id
            })
            .map_or(allocation.alloc_id, |alias| alias.local);
        self.atlases[allocation.region.texture_index].deallocate(
            alloc_id,
            allocation.region.width,
            allocation.region.height,
        );
        self.alloc_id_aliases.retain(|alias| {
            !(alias.texture_index == allocation.region.texture_index
                && alias.original == allocation.alloc_id)
        });
    }
}

/// A layer that has been scheduled and can be sampled by its parent.
#[derive(Debug, Clone, Copy)]
struct ScheduledLayer {
    allocation: LayerAllocation,
    round_idx: usize,
}

/// A layer texture region plus the allocator handle needed to release it.
#[derive(Debug, Clone, Copy)]
struct LayerAllocation {
    region: LayerTextureRegion,
    round_idx: usize,
    alloc_id: AllocId,
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
    fn new(target: RenderTarget, round_idx: usize, initial_load_op: LoadOp) -> Self {
        Self {
            target,
            builder: DrawBuilder::new(target.allows_opaque_pass(), target.geometry_offset()),
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
        let builder = core::mem::replace(
            &mut self.builder,
            DrawBuilder::new(
                self.target.allows_opaque_pass(),
                self.target.geometry_offset(),
            ),
        );
        builder.finish()
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

fn child_layer_ids(cmds: &[RecordedCmd]) -> impl Iterator<Item = u32> + '_ {
    cmds.iter().filter_map(|cmd| match cmd {
        RecordedCmd::Layer(layer_id) => Some(*layer_id),
        RecordedCmd::Batch(_) => None,
    })
}

fn ensure_plain_layer(layer: &vello_common::record::RecordedLayer) -> Result<(), RenderError> {
    if !matches!(layer.kind, RecordedLayerKind::Regular) {
        return Err(RenderError::UnsupportedFeature(
            "filter layers are not supported by schedule_new yet",
        ));
    }
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

#[derive(Debug, Default)]
struct DrawBuilder {
    draw: Draw,
    depth: DepthCounter,
    allow_opaque_pass: bool,
    geometry_offset: (i32, i32),
}

impl DrawBuilder {
    fn new(allow_opaque_pass: bool, geometry_offset: (i32, i32)) -> Self {
        Self {
            draw: Draw::default(),
            depth: DepthCounter::default(),
            allow_opaque_pass,
            geometry_offset,
        }
    }

    fn push_draw(
        &mut self,
        draw: &RecordedDraw,
        strip_storage: &StripStorage,
        scene: &Scene,
        encoded_paints: &[EncodedPaint],
        paint_idxs: &[u32],
    ) {
        match draw {
            RecordedDraw::Path(path) => {
                self.push_path(
                    path.strips.clone(),
                    path.paint.clone(),
                    strip_storage,
                    scene,
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
        scene: &Scene,
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

            if strip.x >= scene.width {
                continue;
            }

            let next_strip = &strips[i + 1];
            let col = strip.alpha_idx() / u32::from(Tile::HEIGHT);
            let next_col = next_strip.alpha_idx() / u32::from(Tile::HEIGHT);
            let strip_width = next_col.saturating_sub(col) as u16;
            let x0 = strip.x;
            let y = strip.y;
            let target_x0 = offset_coord(x0, self.geometry_offset.0);
            let target_y = offset_coord(y, self.geometry_offset.1);

            if strip_width > 0 {
                let processed =
                    ExistingScheduler::process_paint(&paint, encoded_paints, (x0, y), paint_idxs);
                self.draw.push_alpha(
                    GpuStripBuilder::at_surface(target_x0, target_y, strip_width)
                        .with_sparse(strip_width, col)
                        .paint(processed.payload, processed.paint, depth_index),
                    processed.external_texture_id,
                );
            }

            if next_strip.fill_gap() && strip.strip_y() == next_strip.strip_y() {
                let x1 = x0.saturating_add(strip_width);
                let x2 = next_strip.x.min(scene.width);
                if x2 > x1 {
                    let target_x1 = offset_coord(x1, self.geometry_offset.0);
                    let processed = ExistingScheduler::process_paint(
                        &paint,
                        encoded_paints,
                        (x1, y),
                        paint_idxs,
                    );
                    let strip = GpuStripBuilder::at_surface(target_x1, target_y, x2 - x1).paint(
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
        let is_opaque = self.allow_opaque_pass && is_paint_opaque(&rect.paint, encoded_paints);
        let depth_index = self.depth.next(is_opaque);
        pack_rectangle_into_gpu(
            rect,
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
        source: LayerTextureRegion,
        opacity: f32,
        clip_path: Option<&LayerClip>,
        strip_storage: &StripStorage,
    ) {
        // TODO: Add optimization to not emit strips outside of clip bbox.
        if let Some(clip_path) = clip_path {
            self.push_clipped_layer_ref(source, opacity, clip_path, strip_storage);
            return;
        }

        let depth_index = self.depth.next(false);
        // Layer samples are encoded as image-like rect paints. Geometry is transformed into the
        // target allocation, while the payload points at the source atlas coordinate.
        self.draw.push_alpha(
            make_gpu_rect(
                offset_rect_part(
                    RectPart {
                        x: source.scene_bbox.x0,
                        y: source.scene_bbox.y0,
                        width: source.width as u16,
                        height: source.height as u16,
                        frac: 0,
                    },
                    self.geometry_offset,
                ),
                pack_u16_pair(source.x, source.y),
                layer_paint(opacity),
                depth_index,
            ),
            None,
        );
    }

    // TODO: Deduplicate this with vello cpu.
    fn push_clipped_layer_ref(
        &mut self,
        source: LayerTextureRegion,
        opacity: f32,
        clip_path: &LayerClip,
        strip_storage: &StripStorage,
    ) {
        let strips = &strip_storage.strips[clip_path.strip_range.clone()];
        if strips.len() < 2 || clip_path.bbox.is_empty() {
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
            if y < source.scene_bbox.y0 || y >= source.scene_bbox.y1 {
                continue;
            }

            let strip_width = strip.width_to(next_strip);
            if strip_width > 0 {
                let x0 = strip.x.max(source.scene_bbox.x0);
                let x1 = strip
                    .x
                    .saturating_add(strip_width)
                    .min(source.scene_bbox.x1);
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
                            layer_sample_payload(source, x0, y),
                            paint,
                            depth_index,
                        ),
                        None,
                    );
                }
            }

            if next_strip.fill_gap() && next_strip.y == strip.y {
                let x0 = strip
                    .x
                    .saturating_add(strip_width)
                    .max(source.scene_bbox.x0);
                let x1 = if next_strip.is_sentinel() {
                    source.scene_bbox.x1
                } else {
                    next_strip.x.min(source.scene_bbox.x1)
                };
                if x1 > x0 {
                    self.draw.push_alpha(
                        GpuStripBuilder::at_surface(
                            offset_coord(x0, self.geometry_offset.0),
                            offset_coord(y, self.geometry_offset.1),
                            x1 - x0,
                        )
                        .paint(
                            layer_sample_payload(source, x0, y),
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

#[derive(Debug, Default)]
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
        let processed = ExistingScheduler::process_paint(
            &rect.paint,
            encoded_paints,
            (part.x, part.y),
            paint_idxs,
        );
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

fn layer_sample_payload(source: LayerTextureRegion, x: u16, y: u16) -> u32 {
    let source_x = source.x + u32::from(x - source.scene_bbox.x0);
    let source_y = source.y + u32::from(y - source.scene_bbox.y0);
    pack_u16_pair(source_x, source_y)
}

fn layer_paint(opacity: f32) -> u32 {
    (COLOR_SOURCE_LAYER << 29) | u32::from(opacity_to_u8(opacity))
}

fn opacity_to_u8(opacity: f32) -> u8 {
    (opacity.clamp(0.0, 1.0) * 255.0).round() as u8
}
