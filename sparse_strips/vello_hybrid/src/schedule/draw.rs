// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Draw construction for scheduled strip render passes.

use super::builder::CommandStreamState;
use super::{ExternalTextureRun, LayerTextureRegion};
use crate::GpuStrip;
use crate::paint::{COLOR_SOURCE_LAYER, Paints};
use crate::rect::{RectPart, make_gpu_rect, split_rect};
use crate::scene::RecordedDraw;
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
    pub(super) strips: Vec<GpuStrip>,
    pub(super) external_texture_runs: Vec<ExternalTextureRun>,
}

impl Draw {
    #[inline(always)]
    fn push(&mut self, gpu_strip: GpuStrip, external_texture_id: Option<TextureId>) {
        if let Some(texture_id) = external_texture_id {
            let needs_new_run = self
                .external_texture_runs
                .last()
                .is_none_or(|run| run.texture_id != texture_id);
            if needs_new_run {
                let strips_start = if self.external_texture_runs.is_empty() {
                    0
                } else {
                    self.strips.len()
                };
                self.external_texture_runs.push(ExternalTextureRun {
                    strips_start,
                    texture_id,
                });
            }
        }

        self.strips.push(gpu_strip);
    }

    pub(super) fn is_empty(&self) -> bool {
        self.strips.is_empty()
    }
}

impl Clear for Draw {
    fn clear(&mut self) {
        self.strips.clear();
        self.external_texture_runs.clear();
    }
}

pub(super) type OpaqueStrips = Option<Vec<GpuStrip>>;

pub(super) trait OpaqueStripsExt {
    fn new(enabled: bool) -> Self;
    fn is_enabled(&self) -> bool;
    fn push(&mut self, strip: GpuStrip) -> bool;
    fn reverse(&mut self);
    fn strips(&self) -> Option<&[GpuStrip]>;
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

    fn strips(&self) -> Option<&[GpuStrip]> {
        self.as_deref()
    }
}

#[derive(Debug)]
pub(super) struct DrawBuilder<'a> {
    draw: &'a mut Draw,
    opaque: &'a mut OpaqueStrips,
    depth: &'a mut DepthCounter,
    geometry_offset: (i32, i32),
    draw_bounds: RectU16,
}

impl<'a> DrawBuilder<'a> {
    pub(super) fn new(draw: &'a mut Draw, state: &'a mut CommandStreamState) -> Self {
        let geometry_offset = state.target.geometry_offset();
        let draw_bounds = state.draw_bounds;
        Self {
            draw,
            opaque: &mut state.opaque,
            depth: &mut state.depth,
            geometry_offset,
            draw_bounds,
        }
    }

    pub(super) fn push_draw(
        &mut self,
        draw: &RecordedDraw,
        strip_storage: &StripStorage,
        paints: Paints<'_>,
    ) {
        match draw {
            RecordedDraw::Path(path) => {
                self.push_path(
                    path.strips.clone(),
                    path.paint.clone(),
                    strip_storage,
                    paints,
                );
            }
            RecordedDraw::Rect(rect) => {
                self.push_rect(&rect.rect, &rect.paint, paints);
            }
        }
    }

    fn push_path(
        &mut self,
        strips: core::ops::Range<usize>,
        paint: Paint,
        strip_storage: &StripStorage,
        paints: Paints<'_>,
    ) {
        let strips = &strip_storage.strips[strips];

        if strips.is_empty() {
            return;
        }

        let is_opaque = self.opaque.is_enabled() && paints.is_opaque(&paint);
        let depth_index = self.depth.next(is_opaque);

        let tile_bounds = {
            let expanded = self.draw_bounds.snap_to_tile_coordinates();

            RectU16::new(
                expanded.x0 / Tile::WIDTH,
                expanded.y0 / Tile::HEIGHT,
                expanded.x1 / Tile::WIDTH,
                expanded.y1 / Tile::HEIGHT,
            )
        };

        for_each_fill_segment(strips, tile_bounds, |segment| match segment {
            StripSegment::Alpha(segment) => {
                let processed = paints.pack(&paint, (segment.x0(), segment.y()));
                self.draw.push(
                    self.get_fill_strip_with_packed_paint(
                        *segment,
                        Some(segment.col_idx()),
                        processed.payload,
                        processed.paint,
                        depth_index,
                    ),
                    processed.external_texture_id,
                );
            }
            StripSegment::Fill(segment) => {
                let processed = paints.pack(&paint, (segment.x0(), segment.y()));
                let strip = self.get_fill_strip_with_packed_paint(
                    segment,
                    None,
                    processed.payload,
                    processed.paint,
                    depth_index,
                );
                if !is_opaque || !self.opaque.push(strip) {
                    self.draw.push(strip, processed.external_texture_id);
                }
            }
        });
    }

    fn push_rect(&mut self, rect: &Rect, paint: &Paint, paints: Paints<'_>) {
        let clipped_rect = rect.intersect(self.draw_bounds.as_rect());
        if clipped_rect.is_zero_area() {
            return;
        }

        let is_paint_opaque = self.opaque.is_enabled() && paints.is_opaque(paint);
        let depth_index = self.depth.next(is_paint_opaque);
        self.push_rect_parts(&clipped_rect, paint, paints, depth_index, is_paint_opaque);
    }

    pub(super) fn push_layer_fill(
        &mut self,
        sample: LayerSample,
        opacity: f32,
        clip_path: Option<&LayerClip>,
        strip_storage: &StripStorage,
    ) {
        let sample_bbox = sample.bbox.intersect(self.draw_bounds);
        if sample_bbox.is_empty() {
            return;
        }

        let paint = LayerSample::paint(opacity);
        let Some(clip_path) = clip_path else {
            let depth_index = self.depth.next(false);
            // Layer samples are encoded as image-like rect paints. Geometry is transformed into the
            // target allocation, while the payload points at the source atlas coordinate.
            self.draw.push(
                make_gpu_rect(
                    RectPart {
                        rect: sample_bbox.shift(self.geometry_offset),
                        frac: 0,
                    },
                    sample.payload_at(sample_bbox.x0, sample_bbox.y0),
                    paint,
                    depth_index,
                ),
                None,
            );
            return;
        };

        let strips = &strip_storage.strips[clip_path.strip_range.clone()];
        if strips.len() < 2 || clip_path.bbox.is_empty() {
            return;
        }

        let depth_index = self.depth.next(false);
        let tile_bounds = {
            let expanded = sample_bbox.snap_to_tile_coordinates();
            RectU16::new(
                expanded.x0 / Tile::WIDTH,
                expanded.y0 / Tile::HEIGHT,
                expanded.x1 / Tile::WIDTH,
                expanded.y1 / Tile::HEIGHT,
            )
        };

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
        let strip =
            self.get_fill_strip_with_packed_paint(segment, col_idx, payload, paint, depth_index);
        self.draw.push(strip, None);
    }

    fn get_fill_strip_with_packed_paint(
        &self,
        segment: StripFillSegment,
        col_idx: Option<u32>,
        payload: u32,
        paint: u32,
        depth_index: u32,
    ) -> GpuStrip {
        let rect = segment.shift(self.geometry_offset);
        let width = rect.width();
        let (dense_width_or_rect_height, col_idx_or_rect_frac) = if let Some(col_idx) = col_idx {
            (width, col_idx)
        } else {
            (0, 0)
        };

        GpuStrip {
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

    fn push_rect_parts(
        &mut self,
        rect: &Rect,
        paint: &Paint,
        paints: Paints<'_>,
        depth_index: u32,
        is_paint_opaque: bool,
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
            let processed = paints.pack(paint, (part.rect.x0, part.rect.y0));
            let strip = make_gpu_rect(
                part.shift(self.geometry_offset),
                processed.payload,
                processed.paint,
                depth_index,
            );
            if !(is_first && is_paint_opaque && part.frac == 0 && self.opaque.push(strip)) {
                self.draw.push(strip, processed.external_texture_id);
            }
            is_first = false;
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
