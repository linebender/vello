// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Schedule construction for the new hybrid scheduler.

use super::round::{Round, Schedule};
use super::{Draw, LayerTextureRegion, RenderTarget, RootRenderTarget};
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
use vello_common::peniko::BlendMode;
use vello_common::record::{RecordedCmd, RecordedLayerKind};
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
    layer_atlases: [Atlas; 2],
    layer_allocations: Vec<Option<ScheduledLayer>>,
    pending_deallocations: Vec<Vec<LayerAllocation>>,
    current_round: usize,
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
        Self {
            scene,
            strip_storage,
            root_output_target,
            paint_idxs,
            encoded_paints,
            layer_atlases: [
                Atlas::new(AtlasId::new(0), layer_texture_size.0, layer_texture_size.1),
                Atlas::new(AtlasId::new(1), layer_texture_size.0, layer_texture_size.1),
            ],
            layer_allocations: alloc::vec![None; layer_count],
            pending_deallocations: alloc::vec![Vec::new()],
            current_round: 0,
            layer_texture_size,
        }
    }

    pub(super) fn build(&mut self) -> Result<Schedule, RenderError> {
        self.validate_layers()?;

        let mut schedule = Schedule {
            rounds: alloc::vec![Round::default()],
        };

        // Walk the command tree left-to-right. Each encountered layer schedules its own subtree
        // first, then materializes itself into the earliest round that can sample those children.
        self.schedule_children(&self.scene.recorder.root_cmds, &mut schedule)?;
        self.schedule_root(&mut schedule);

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
        if layer.bbox.is_empty() {
            return Ok(());
        }

        let texture_index = layer.depth & 1;
        let target_round = self.earliest_round_for_target(cmds, Some(texture_index));
        let allocation =
            self.allocate_layer(layer_id, texture_index, layer.bbox, target_round, schedule)?;
        let target = RenderTarget::Layer(allocation.allocation.region);
        let draw = self.build_draw(cmds, target);

        if !draw.is_empty() {
            schedule.rounds[allocation.round_idx].push_pass(target, draw);
        }

        // Once this layer has sampled its children, their atlas regions can be cleared after the
        // round completes. The allocator release is delayed until we advance past that round.
        self.consume_sampled_children(cmds, allocation.round_idx, schedule);

        Ok(())
    }

    fn schedule_root(&mut self, schedule: &mut Schedule) {
        let round_idx = self.earliest_round_for_target(&self.scene.recorder.root_cmds, None);
        self.advance_to_round(round_idx, schedule);

        let target = RenderTarget::Root(self.root_output_target);
        let draw = self.build_draw(&self.scene.recorder.root_cmds, target);
        if !draw.is_empty() {
            schedule.rounds[round_idx].push_pass(target, draw);
        }

        // Root sampling is the final consumer for any direct child layers.
        self.consume_sampled_children(&self.scene.recorder.root_cmds, round_idx, schedule);
    }

    fn build_draw(&self, cmds: &[RecordedCmd], target: RenderTarget) -> Draw {
        let mut builder = DrawBuilder::new(target.allows_opaque_pass(), target.geometry_offset());
        for cmd in cmds {
            match cmd {
                RecordedCmd::Batch(range) => {
                    for draw in &self.scene.recorder.draws[range.start as usize..range.end as usize]
                    {
                        builder.push_draw(
                            draw,
                            self.strip_storage,
                            self.scene,
                            self.encoded_paints,
                            self.paint_idxs,
                        );
                    }
                }
                RecordedCmd::Layer(layer_id) => {
                    if let Some(layer) = self.layer_allocations[*layer_id as usize] {
                        builder.push_layer_ref(layer.allocation.region);
                    }
                }
            }
        }

        builder.finish()
    }

    fn consume_sampled_children(
        &mut self,
        cmds: &[RecordedCmd],
        round_idx: usize,
        schedule: &mut Schedule,
    ) {
        let mut consumed_layers = Vec::new();
        for layer_id in child_layer_ids(cmds) {
            if consumed_layers.contains(&layer_id) {
                continue;
            }
            consumed_layers.push(layer_id);

            let Some(scheduled_layer) = self.layer_allocations[layer_id as usize].take() else {
                continue;
            };

            schedule.rounds[round_idx]
                .clear_layer_regions
                .push(scheduled_layer.allocation.region);
            self.pending_deallocations[round_idx].push(scheduled_layer.allocation);
        }
    }

    fn allocate_layer(
        &mut self,
        layer_id: u32,
        texture_index: usize,
        bbox: RectU16,
        earliest_round: usize,
        schedule: &mut Schedule,
    ) -> Result<ScheduledLayer, RenderError> {
        let width = u32::from(bbox.width());
        let height = u32::from(bbox.height());
        if width > self.layer_texture_size.0 || height > self.layer_texture_size.1 {
            return Err(RenderError::AtlasError(AtlasError::NoSpaceAvailable));
        }

        let mut round_idx = earliest_round;
        loop {
            // Advancing applies all completed-round deallocations, so a retry can reuse memory
            // freed by parent samples in earlier rounds.
            self.advance_to_round(round_idx, schedule);

            if let Some(allocation) = self.layer_atlases[texture_index].allocate(width, height) {
                let scheduled_layer = ScheduledLayer {
                    allocation: LayerAllocation {
                        region: LayerTextureRegion {
                            texture_index,
                            x: allocation.x,
                            y: allocation.y,
                            width,
                            height,
                            scene_bbox: bbox,
                        },
                        alloc_id: allocation.id,
                    },
                    round_idx,
                };
                self.layer_allocations[layer_id as usize] = Some(scheduled_layer);
                return Ok(scheduled_layer);
            }

            // If nothing is due to be freed after this round, another retry would see the same
            // allocator state and would spin forever.
            if self
                .pending_deallocations
                .get(round_idx)
                .is_none_or(Vec::is_empty)
            {
                return Err(RenderError::AtlasError(AtlasError::NoSpaceAvailable));
            }

            round_idx += 1;
        }
    }

    fn earliest_round_for_target(
        &self,
        cmds: &[RecordedCmd],
        target_texture_index: Option<usize>,
    ) -> usize {
        let mut round_idx = self.current_round;
        for layer_id in child_layer_ids(cmds) {
            let Some(child) = self.layer_allocations[layer_id as usize] else {
                continue;
            };

            // A layer may sample a child in the same round only when the fixed texture ordering
            // guarantees that the child texture is rendered before the parent texture.
            let child_texture_index = child.allocation.region.texture_index;
            let child_round = child.round_idx;
            let required_round = match target_texture_index {
                Some(target_texture_index)
                    if can_sample_layer_in_same_round(
                        target_texture_index,
                        child_texture_index,
                    ) =>
                {
                    child_round
                }
                Some(_) => child_round + 1,
                None => child_round,
            };
            round_idx = round_idx.max(required_round);
        }

        round_idx
    }

    fn advance_to_round(&mut self, round_idx: usize, schedule: &mut Schedule) {
        self.ensure_round_exists(round_idx, schedule);
        while self.current_round < round_idx {
            let finished_round = self.current_round;
            // Clear commands are stored on the finished round, but atlas memory becomes reusable
            // only now, before the next round starts.
            for allocation in self.pending_deallocations[finished_round].drain(..) {
                self.layer_atlases[allocation.region.texture_index].deallocate(
                    allocation.alloc_id,
                    allocation.region.width,
                    allocation.region.height,
                );
            }
            self.current_round += 1;
        }
    }

    fn ensure_round_exists(&mut self, round_idx: usize, schedule: &mut Schedule) {
        while schedule.rounds.len() <= round_idx {
            schedule.rounds.push(Round::default());
        }
        while self.pending_deallocations.len() <= round_idx {
            self.pending_deallocations.push(Vec::new());
        }
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
    alloc_id: AllocId,
}

fn can_sample_layer_in_same_round(parent_texture_index: usize, child_texture_index: usize) -> bool {
    parent_texture_index != child_texture_index
        && layer_texture_order(child_texture_index) < layer_texture_order(parent_texture_index)
}

fn layer_texture_order(texture_index: usize) -> usize {
    match texture_index {
        1 => 0,
        0 => 1,
        _ => texture_index,
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
    if layer.props.blend_mode != BlendMode::default() {
        return Err(RenderError::UnsupportedFeature(
            "non-default blend layers are not supported by schedule_new yet",
        ));
    }
    if layer.props.opacity != 1.0 {
        return Err(RenderError::UnsupportedFeature(
            "opacity layers are not supported by schedule_new yet",
        ));
    }
    if layer.props.mask.is_some() {
        return Err(RenderError::UnsupportedFeature(
            "mask layers are not supported by schedule_new yet",
        ));
    }
    if layer.props.clip_path.is_some() {
        return Err(RenderError::UnsupportedFeature(
            "clip layers are not supported by schedule_new yet",
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

    fn push_layer_ref(&mut self, source: LayerTextureRegion) {
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
                COLOR_SOURCE_LAYER << 29,
                depth_index,
            ),
            None,
        );
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
