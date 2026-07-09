// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Draw construction for scheduled strip render passes.

use super::CommandStreamState;
use super::buffer::{Ranges, ScheduleBuffers, VecExt};
use super::{ExternalTextureRun, LayerTextureRegion};
use crate::GpuStrip;
use crate::paint::{COLOR_SOURCE_LAYER, PaintResolver};
use crate::rect::{RectPart, split_rect};
use crate::scene::{RecordedDraw, RecordedPath};
use crate::util::{pack_opacity, pack_u16_pair};
use ::alloc::vec::Vec;
use vello_common::TextureId;
use vello_common::geometry::RectU16;
use vello_common::kurbo::Rect;
use vello_common::paint::Paint;
use vello_common::record::LayerClip;
use vello_common::strip::{StripFillSegment, StripSegment, for_each_fill_segment};
use vello_common::strip_generator::StripStorage;
use vello_common::tile::Tile;
use vello_common::util::{Clear, RectExt};

#[derive(Debug, Default, Clone)]
pub(super) struct Draw {
    pub(super) strip_ranges: Ranges,
    pub(super) external_texture_runs: Vec<ExternalTextureRun>,
}

impl Draw {
    #[inline]
    fn push(
        &mut self,
        buffers: &mut ScheduleBuffers,
        gpu_strip: GpuStrip,
        external_texture_id: Option<TextureId>,
    ) {
        if let Some(texture_id) = external_texture_id {
            let needs_new_run = self
                .external_texture_runs
                .last()
                .is_none_or(|run| run.texture_id != texture_id);

            if needs_new_run {
                let strips_start = if self.external_texture_runs.is_empty() {
                    0
                } else {
                    self.strip_ranges.len()
                };

                self.external_texture_runs.push(ExternalTextureRun {
                    strips_start,
                    texture_id,
                });
            }
        }

        buffers
            .strips
            .push_ranged(&mut self.strip_ranges, gpu_strip);
    }
}

impl Clear for Draw {
    fn clear(&mut self) {
        self.strip_ranges.clear();
        self.external_texture_runs.clear();
    }
}

pub(super) type OpaqueStrips = Option<Vec<GpuStrip>>;

pub(super) trait OpaqueStripsExt {
    fn new(enabled: bool) -> Self;
    fn is_enabled(&self) -> bool;
    fn push(&mut self, strip: GpuStrip) -> bool;
    fn reverse(&mut self);
}

impl OpaqueStripsExt for OpaqueStrips {
    fn new(enabled: bool) -> Self {
        enabled.then(Vec::new)
    }

    fn is_enabled(&self) -> bool {
        self.is_some()
    }

    fn push(&mut self, strip: GpuStrip) -> bool {
        let Some(strips) = self else {
            return false;
        };

        strips.push(strip);
        true
    }

    fn reverse(&mut self) {
        if let Some(strips) = self {
            strips.reverse();
        }
    }
}

#[derive(Debug)]
pub(super) struct DrawBuilder<'a> {
    draw: &'a mut Draw,
    buffers: &'a mut ScheduleBuffers,
    state: &'a mut CommandStreamState,
}

impl<'a> DrawBuilder<'a> {
    pub(super) fn new(
        draw: &'a mut Draw,
        buffers: &'a mut ScheduleBuffers,
        state: &'a mut CommandStreamState,
    ) -> Self {
        Self {
            draw,
            buffers,
            state,
        }
    }

    pub(super) fn push_draw(
        &mut self,
        draw: &RecordedDraw,
        strip_storage: &StripStorage,
        paint_resolver: PaintResolver<'_>,
    ) {
        match draw {
            RecordedDraw::Path(path) => self.push_path(path, strip_storage, paint_resolver),
            RecordedDraw::Rect(rect) => {
                self.push_rect(&rect.rect, &rect.paint, paint_resolver);
            }
        }
    }

    fn push_path(
        &mut self,
        path: &RecordedPath,
        strip_storage: &StripStorage,
        paint_resolver: PaintResolver<'_>,
    ) {
        let strips = &strip_storage.strips[path.strips.clone()];

        let paint = paint_resolver.pack(&path.paint);
        let is_opaque = self.state.opaque.is_enabled() && paint.opaque;
        let depth_index = self.state.depth.next(is_opaque);
        let tile_bounds = self.state.draw_bounds.to_tile_bounds();

        // Note: This method will also take care of culling any strips to the active clip bbox.
        for_each_fill_segment(strips, tile_bounds, |segment| match segment {
            StripSegment::Alpha(segment) => {
                let shifted = segment.shift(self.state.target.geometry_offset());
                let strip = GpuStrip::from_fill(
                    shifted,
                    Some(segment.col_idx()),
                    paint.payload_at(segment.x0(), segment.y()),
                    paint.paint,
                    depth_index,
                );

                self.draw
                    .push(self.buffers, strip, paint.external_texture_id);
            }
            StripSegment::Fill(segment) => {
                let shifted = segment.shift(self.state.target.geometry_offset());

                let strip = GpuStrip::from_fill(
                    shifted,
                    None,
                    paint.payload_at(segment.x0(), segment.y()),
                    paint.paint,
                    depth_index,
                );

                if !is_opaque || !self.state.opaque.push(strip) {
                    self.draw
                        .push(self.buffers, strip, paint.external_texture_id);
                }
            }
        });
    }

    fn push_rect(&mut self, rect: &Rect, paint: &Paint, paint_resolver: PaintResolver<'_>) {
        // Recordings might contain geometry that exceeds the actual layer
        // bounding box. This can happen when a clip path is associated with the layer.
        // Recordings will not cull those for us, so we need to do this manually here.
        // For normal paths, the `for_each_fill_segment` method takes care of doing this.
        // For rectangles, we can do a simple intersection.
        let clipped_rect = rect.intersect(self.state.draw_bounds.as_rect());
        if clipped_rect.is_zero_area() {
            return;
        }

        let paint = paint_resolver.pack(paint);
        let is_paint_opaque = self.state.opaque.is_enabled() && paint.opaque;
        let depth_index = self.state.depth.next(is_paint_opaque);

        let split = split_rect(&clipped_rect);

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
            let shifted = part.shift(self.state.target.geometry_offset());

            let strip = GpuStrip::from_rect(
                shifted,
                paint.payload_at(part.rect.x0, part.rect.y0),
                paint.paint,
                depth_index,
            );

            if !(is_paint_opaque && part.frac == 0 && self.state.opaque.push(strip)) {
                self.draw
                    .push(self.buffers, strip, paint.external_texture_id);
            }
        }
    }

    pub(super) fn push_layer_fill(
        &mut self,
        sample: LayerSample,
        opacity: f32,
        clip_path: Option<&LayerClip>,
        strip_storage: &StripStorage,
    ) {
        let sample_bbox = sample.bbox.intersect(self.state.draw_bounds);
        if sample_bbox.is_empty() {
            return;
        }

        let paint = LayerSample::paint(opacity);
        if let Some(clip_path) = clip_path {
            let strips = &strip_storage.strips[clip_path.strip_range.clone()];
            let depth_index = self.state.depth.next(false);
            let tile_bounds = sample_bbox.to_tile_bounds();

            for_each_fill_segment(strips, tile_bounds, |segment| match segment {
                StripSegment::Alpha(segment) => self.push_layer_fill_segment(
                    sample,
                    *segment,
                    Some(segment.col_idx()),
                    paint,
                    depth_index,
                ),
                StripSegment::Fill(segment) => {
                    self.push_layer_fill_segment(sample, segment, None, paint, depth_index);
                }
            });
        } else {
            let depth_index = self.state.depth.next(false);

            let rect_part = RectPart {
                rect: sample_bbox.shift(self.state.target.geometry_offset()),
                frac: 0,
            };

            self.draw.push(
                self.buffers,
                GpuStrip::from_rect(
                    rect_part,
                    sample.payload_at(sample_bbox.x0, sample_bbox.y0),
                    paint,
                    depth_index,
                ),
                None,
            );
        }
    }

    fn push_layer_fill_segment(
        &mut self,
        sample: LayerSample,
        segment: StripFillSegment,
        col_idx: Option<u32>,
        paint: u32,
        depth_index: u32,
    ) {
        let payload = sample.payload_at(segment.x0(), segment.y());
        let shifted = segment.shift(self.state.target.geometry_offset());
        let strip = GpuStrip::from_fill(shifted, col_idx, payload, paint, depth_index);
        self.draw.push(self.buffers, strip, None);
    }
}

/// Bit 31 of [`GpuStrip::paint_and_rect_flag`] signals that the strip
/// represents a full rectangle.
const RECT_STRIP_FLAG: u32 = 1 << 31;

impl GpuStrip {
    fn from_fill(
        rect: RectU16,
        col_idx: Option<u32>,
        payload: u32,
        paint: u32,
        depth_index: u32,
    ) -> Self {
        let width = rect.width();
        let (dense_width_or_rect_height, col_idx_or_rect_frac) = if let Some(col_idx) = col_idx {
            (width, col_idx)
        } else {
            (0, 0)
        };

        Self {
            x: rect.x0,
            y: rect.y0,
            width,
            dense_width_or_rect_height,
            col_idx_or_rect_frac,
            payload,
            paint_and_rect_flag: paint,
            depth_index,
        }
    }

    fn from_rect(part: RectPart, payload: u32, paint: u32, depth_index: u32) -> Self {
        Self {
            x: part.rect.x0,
            y: part.rect.y0,
            width: part.rect.width(),
            dense_width_or_rect_height: part.rect.height(),
            col_idx_or_rect_frac: part.frac,
            payload,
            paint_and_rect_flag: paint | RECT_STRIP_FLAG,
            depth_index,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct LayerSample {
    pub(super) source: LayerTextureRegion,
    pub(super) bbox: RectU16,
    pub(super) source_origin: (u16, u16),
}

impl LayerSample {
    fn payload_at(self, x: u16, y: u16) -> u32 {
        let source = self.source;
        let source_x = source.texture.rect.x0 + self.source_origin.0 + (x - source.scene_bbox.x0);
        let source_y = source.texture.rect.y0 + self.source_origin.1 + (y - source.scene_bbox.y0);
        pack_u16_pair(source_x, source_y)
    }

    fn paint(opacity: f32) -> u32 {
        (COLOR_SOURCE_LAYER << 29) | u32::from(pack_opacity(opacity))
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct DepthCounter {
    count: u32,
}

impl DepthCounter {
    #[inline(always)]
    fn next(&mut self, opaque: bool) -> u32 {
        self.count += opaque as u32;

        self.count
    }
}

trait RectU16Ext {
    fn to_tile_bounds(self) -> RectU16;
}

impl RectU16Ext for RectU16 {
    fn to_tile_bounds(self) -> RectU16 {
        let bounds = self.snap_to_tile_coordinates();

        Self::new(
            bounds.x0 / Tile::WIDTH,
            bounds.y0 / Tile::HEIGHT,
            bounds.x1 / Tile::WIDTH,
            bounds.y1 / Tile::HEIGHT,
        )
    }
}
