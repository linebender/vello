// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Draw construction for scheduled strip render passes.

use super::{ExternalTextureRun, LayerTextureRegion};
use crate::GpuStrip;
use crate::rect::{RectPart, make_gpu_rect, split_rect};
use crate::scene::RecordedDraw;
use crate::util::{pack_opacity, pack_u16_pair};
use ::alloc::vec::Vec;
use vello_common::TextureId;
use vello_common::encode::{EncodedKind, EncodedPaint};
use vello_common::geometry::RectU16;
use vello_common::kurbo::Rect;
use vello_common::paint::{ImageSource, Paint};
use vello_common::record::LayerClip;
use vello_common::strip::{StripSegment, for_each_fill_segment};
use vello_common::strip_generator::StripStorage;
use vello_common::tile::Tile;

const COLOR_SOURCE_PAYLOAD: u32 = 0;
const COLOR_SOURCE_LAYER: u32 = 1;

const PAINT_TYPE_SOLID: u32 = 0;
const PAINT_TYPE_IMAGE: u32 = 1;
const PAINT_TYPE_LINEAR_GRADIENT: u32 = 2;
const PAINT_TYPE_RADIAL_GRADIENT: u32 = 3;
const PAINT_TYPE_SWEEP_GRADIENT: u32 = 4;
const PAINT_TYPE_BLURRED_ROUNDED_RECT: u32 = 5;

#[derive(Clone, Copy)]
struct PackedPaint {
    payload: u32,
    paint: u32,
    external_texture_id: Option<TextureId>,
}

/// Process a paint and return the packed payload, paint and optional external texture id.
#[inline(always)]
fn pack_paint(
    paint: &Paint,
    encoded_paints: &[EncodedPaint],
    (scene_strip_x, scene_strip_y): (u16, u16),
    paint_idxs: &[u32],
) -> PackedPaint {
    match paint {
        Paint::Solid(color) => {
            let rgba = color.as_premul_rgba8().to_u32();
            let paint_packed = (COLOR_SOURCE_PAYLOAD << 30) | (PAINT_TYPE_SOLID << 27);
            PackedPaint {
                payload: rgba,
                paint: paint_packed,
                external_texture_id: None,
            }
        }
        Paint::Indexed(indexed_paint) => {
            let paint_id = indexed_paint.index();
            let paint_idx = paint_idxs.get(paint_id).copied().unwrap();

            let Some(encoded_paint) = encoded_paints.get(paint_id) else {
                unimplemented!("Unsupported paint type");
            };

            match encoded_paint {
                EncodedPaint::Image(encoded_image) => match &encoded_image.source {
                    ImageSource::OpaqueId { .. } => {
                        let paint_packed = (COLOR_SOURCE_PAYLOAD << 29)
                            | (PAINT_TYPE_IMAGE << 26)
                            | (paint_idx & 0x03FF_FFFF);
                        let scene_strip_xy =
                            ((scene_strip_y as u32) << 16) | (scene_strip_x as u32);
                        PackedPaint {
                            payload: scene_strip_xy,
                            paint: paint_packed,
                            external_texture_id: None,
                        }
                    }
                    _ => unimplemented!("Unsupported image source"),
                },
                EncodedPaint::ExternalTexture(texture) => {
                    let paint_packed = (COLOR_SOURCE_PAYLOAD << 29)
                        | (PAINT_TYPE_IMAGE << 26)
                        | (paint_idx & 0x03FF_FFFF);
                    let scene_strip_xy = ((scene_strip_y as u32) << 16) | (scene_strip_x as u32);
                    PackedPaint {
                        payload: scene_strip_xy,
                        paint: paint_packed,
                        external_texture_id: Some(texture.texture_id),
                    }
                }
                EncodedPaint::Gradient(gradient) => {
                    let gradient_paint_type = match &gradient.kind {
                        EncodedKind::Linear(_) => PAINT_TYPE_LINEAR_GRADIENT,
                        EncodedKind::Radial(_) => PAINT_TYPE_RADIAL_GRADIENT,
                        EncodedKind::Sweep(_) => PAINT_TYPE_SWEEP_GRADIENT,
                    };
                    let paint_packed = (COLOR_SOURCE_PAYLOAD << 29)
                        | (gradient_paint_type << 26)
                        | (paint_idx & 0x03FF_FFFF);
                    let scene_strip_xy = ((scene_strip_y as u32) << 16) | (scene_strip_x as u32);
                    PackedPaint {
                        payload: scene_strip_xy,
                        paint: paint_packed,
                        external_texture_id: None,
                    }
                }
                EncodedPaint::BlurredRoundedRect(_) => {
                    let paint_packed = (COLOR_SOURCE_PAYLOAD << 29)
                        | (PAINT_TYPE_BLURRED_ROUNDED_RECT << 26)
                        | (paint_idx & 0x03FF_FFFF);
                    let scene_strip_xy = ((scene_strip_y as u32) << 16) | (scene_strip_x as u32);
                    PackedPaint {
                        payload: scene_strip_xy,
                        paint: paint_packed,
                        external_texture_id: None,
                    }
                }
            }
        }
    }
}

#[derive(Debug, Default, Clone)]
pub(super) struct Draw {
    pub(super) alpha: Vec<GpuStrip>,
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
                    self.alpha.len()
                };
                self.external_texture_runs.push(ExternalTextureRun {
                    strips_start,
                    texture_id,
                });
            }
        }

        self.alpha.push(gpu_strip);
    }

    pub(super) fn is_empty(&self) -> bool {
        self.alpha.is_empty()
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct LayerSample {
    pub(super) source: LayerTextureRegion,
    pub(super) bbox: RectU16,
    pub(super) source_origin: (u16, u16),
}

#[derive(Debug)]
pub(super) struct DrawBuilder<'a> {
    draw: Draw,
    opaque: Option<&'a mut Vec<GpuStrip>>,
    opaque_start: usize,
    depth: &'a mut DepthCounter,
    geometry_offset: (i32, i32),
    draw_bounds: RectU16,
}

impl<'a> DrawBuilder<'a> {
    pub(super) fn new(
        opaque: Option<&'a mut Vec<GpuStrip>>,
        depth: &'a mut DepthCounter,
        geometry_offset: (i32, i32),
        draw_bounds: RectU16,
    ) -> Self {
        let opaque_start = opaque.as_ref().map_or(0, |opaque| opaque.len());
        Self {
            draw: Draw::default(),
            opaque,
            opaque_start,
            depth,
            geometry_offset,
            draw_bounds,
        }
    }

    pub(super) fn push_draw(
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
                self.push_rect(&rect.rect, &rect.paint, encoded_paints, paint_idxs);
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

        let is_opaque = self.opaque.is_some() && is_paint_opaque(&paint, encoded_paints);
        let depth_index = self.depth.next(is_opaque);

        for_each_fill_segment(
            strips,
            tile_bounds(self.draw_bounds),
            |segment| match segment {
                StripSegment::Alpha(segment) => {
                    let x0 = segment.tile_x0 * Tile::WIDTH;
                    let x1 = segment.tile_x1 * Tile::WIDTH;
                    let y = segment.tile_y * Tile::HEIGHT;
                    let processed = pack_paint(&paint, encoded_paints, (x0, y), paint_idxs);
                    self.draw.push(
                        self.get_fill_strip(
                            x0,
                            x1,
                            y,
                            Some(segment.alpha_idx / u32::from(Tile::HEIGHT)),
                            &processed,
                            depth_index,
                        ),
                        processed.external_texture_id,
                    );
                }
                StripSegment::Fill(segment) => {
                    let x0 = segment.tile_x0 * Tile::WIDTH;
                    let x1 = segment.tile_x1 * Tile::WIDTH;
                    let y = segment.tile_y * Tile::HEIGHT;
                    let processed = pack_paint(&paint, encoded_paints, (x0, y), paint_idxs);
                    let strip = self.get_fill_strip(x0, x1, y, None, &processed, depth_index);
                    if !is_opaque || !self.push_opaque(strip) {
                        self.draw.push(strip, processed.external_texture_id);
                    }
                }
            },
        );
    }

    fn push_rect(
        &mut self,
        rect: &Rect,
        paint: &Paint,
        encoded_paints: &[EncodedPaint],
        paint_idxs: &[u32],
    ) {
        let Some(rect) = clip_rect_to_draw_bounds(rect, self.draw_bounds) else {
            return;
        };
        let is_paint_opaque = self.opaque.is_some() && is_paint_opaque(paint, encoded_paints);
        let depth_index = self.depth.next(is_paint_opaque);
        pack_rectangle_into_gpu(
            &rect,
            paint,
            encoded_paints,
            paint_idxs,
            depth_index,
            is_paint_opaque,
            self.geometry_offset,
            &mut self.opaque,
            &mut self.draw,
        );
    }

    pub(super) fn push_layer_ref(
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
        let bbox = sample.bbox.intersect(self.draw_bounds);
        if bbox.is_empty() {
            return;
        }

        // Layer samples are encoded as image-like rect paints. Geometry is transformed into the
        // target allocation, while the payload points at the source atlas coordinate.
        self.draw.push(
            make_gpu_rect(
                RectPart {
                    rect: bbox.shift(self.geometry_offset),
                    frac: 0,
                },
                layer_sample_payload(sample, bbox.x0, bbox.y0),
                layer_paint(opacity),
                depth_index,
            ),
            None,
        );
    }

    fn push_clipped_layer_ref(
        &mut self,
        sample: LayerSample,
        opacity: f32,
        clip_path: &LayerClip,
        strip_storage: &StripStorage,
    ) {
        let strips = &strip_storage.strips[clip_path.strip_range.clone()];
        let sample_bbox = sample.bbox.intersect(self.draw_bounds);
        if strips.len() < 2 || clip_path.bbox.is_empty() || sample_bbox.is_empty() {
            return;
        }

        let depth_index = self.depth.next(false);
        let paint = layer_paint(opacity);

        for_each_fill_segment(strips, tile_bounds(sample_bbox), |segment| match segment {
            StripSegment::Alpha(segment) => {
                let x0 = segment.tile_x0 * Tile::WIDTH;
                let x1 = segment.tile_x1 * Tile::WIDTH;
                let y = segment.tile_y * Tile::HEIGHT;
                self.draw.push(
                    self.get_layer_fill_strip(
                        sample,
                        x0,
                        x1,
                        y,
                        Some(segment.alpha_idx / u32::from(Tile::HEIGHT)),
                        paint,
                        depth_index,
                    ),
                    None,
                );
            }
            StripSegment::Fill(segment) => {
                let x0 = segment.tile_x0 * Tile::WIDTH;
                let x1 = segment.tile_x1 * Tile::WIDTH;
                let y = segment.tile_y * Tile::HEIGHT;
                self.draw.push(
                    self.get_layer_fill_strip(sample, x0, x1, y, None, paint, depth_index),
                    None,
                );
            }
        });
    }

    fn get_fill_strip(
        &self,
        x0: u16,
        x1: u16,
        y: u16,
        col_idx: Option<u32>,
        paint: &PackedPaint,
        depth_index: u32,
    ) -> GpuStrip {
        self.get_fill_strip_with_packed_paint(
            x0,
            x1,
            y,
            col_idx,
            paint.payload,
            paint.paint,
            depth_index,
        )
    }

    fn get_layer_fill_strip(
        &self,
        sample: LayerSample,
        x0: u16,
        x1: u16,
        y: u16,
        col_idx: Option<u32>,
        paint: u32,
        depth_index: u32,
    ) -> GpuStrip {
        self.get_fill_strip_with_packed_paint(
            x0,
            x1,
            y,
            col_idx,
            layer_sample_payload(sample, x0, y),
            paint,
            depth_index,
        )
    }

    fn get_fill_strip_with_packed_paint(
        &self,
        x0: u16,
        x1: u16,
        y: u16,
        col_idx: Option<u32>,
        payload: u32,
        paint: u32,
        depth_index: u32,
    ) -> GpuStrip {
        let x = offset_coord(x0, self.geometry_offset.0);
        let y = offset_coord(y, self.geometry_offset.1);
        let width = x1 - x0;
        let (dense_width_or_rect_height, col_idx_or_rect_frac) = if let Some(col_idx) = col_idx {
            (width, col_idx)
        } else {
            (0, 0)
        };

        GpuStrip {
            x,
            y,
            width,
            dense_width_or_rect_height,
            col_idx_or_rect_frac,
            payload,
            paint_and_rect_flag: paint,
            depth_index,
        }
    }

    pub(super) fn append_to(&mut self, target: &mut Draw) {
        self.finish_opaque_segment();
        self.append_alpha_to(target);
    }

    fn push_opaque(&mut self, strip: GpuStrip) -> bool {
        let Some(opaque) = self.opaque.as_deref_mut() else {
            return false;
        };
        opaque.push(strip);
        true
    }

    fn finish_opaque_segment(&mut self) {
        if let Some(opaque) = self.opaque.as_deref_mut() {
            opaque[self.opaque_start..].reverse();
            self.opaque_start = opaque.len();
        }
    }

    fn append_alpha_to(&mut self, target: &mut Draw) {
        let alpha_offset = target.alpha.len();
        let had_external_texture_runs = !target.external_texture_runs.is_empty();
        for (idx, run) in self.draw.external_texture_runs.drain(..).enumerate() {
            let strips_start = if !had_external_texture_runs && idx == 0 {
                0
            } else {
                alpha_offset + run.strips_start
            };
            if strips_start == target.alpha.len()
                && target
                    .external_texture_runs
                    .last()
                    .is_some_and(|last| last.texture_id == run.texture_id)
            {
                continue;
            }
            target.external_texture_runs.push(ExternalTextureRun {
                texture_id: run.texture_id,
                strips_start,
            });
        }
        target.alpha.append(&mut self.draw.alpha);
    }

    pub(super) fn is_empty(&self) -> bool {
        let opaque_empty = self
            .opaque
            .as_ref()
            .is_none_or(|opaque| opaque.len() == self.opaque_start);
        opaque_empty && self.draw.is_empty()
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
    rect: &Rect,
    paint: &Paint,
    encoded_paints: &[EncodedPaint],
    paint_idxs: &[u32],
    depth_index: u32,
    is_paint_opaque: bool,
    geometry_offset: (i32, i32),
    opaque: &mut Option<&mut Vec<GpuStrip>>,
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
        let processed = pack_paint(
            paint,
            encoded_paints,
            (part.rect.x0, part.rect.y0),
            paint_idxs,
        );
        let strip = make_gpu_rect(
            part.shift(geometry_offset),
            processed.payload,
            processed.paint,
            depth_index,
        );
        if is_first
            && is_paint_opaque
            && part.frac == 0
            && let Some(opaque) = opaque.as_deref_mut()
        {
            opaque.push(strip);
        } else {
            draw.push(strip, processed.external_texture_id);
        }
        is_first = false;
    }
}

fn clip_rect_to_draw_bounds(rect: &Rect, bounds: RectU16) -> Option<Rect> {
    // Path draws are clipped by `for_each_fill_segment(..., tile_bounds(draw_bounds), ...)`.
    // Fast rects bypass strip iteration, so they need the equivalent target-local clip here
    // before applying the layer atlas geometry offset.
    let bounds = Rect::new(
        f64::from(bounds.x0),
        f64::from(bounds.y0),
        f64::from(bounds.x1),
        f64::from(bounds.y1),
    );
    let rect = rect.intersect(bounds);

    (rect.x0 < rect.x1 && rect.y0 < rect.y1).then_some(rect)
}

fn tile_bounds(bounds: RectU16) -> RectU16 {
    RectU16::new(
        bounds.x0 / Tile::WIDTH,
        bounds.y0 / Tile::HEIGHT,
        bounds.x1.div_ceil(Tile::WIDTH),
        bounds.y1.div_ceil(Tile::HEIGHT),
    )
}

fn offset_coord(coord: u16, offset: i32) -> u16 {
    let coord = i32::from(coord) + offset;
    debug_assert!(
        (0..=i32::from(u16::MAX)).contains(&coord),
        "offset coordinate must fit into u16"
    );
    u16::try_from(coord).expect("offset coordinate must fit into u16")
}

fn layer_sample_payload(sample: LayerSample, x: u16, y: u16) -> u32 {
    let source = sample.source;
    let source_x = source.texture.rect.x0 + sample.source_origin.0 + (x - source.scene_bbox.x0);
    let source_y = source.texture.rect.y0 + sample.source_origin.1 + (y - source.scene_bbox.y0);
    pack_u16_pair(source_x, source_y)
}

fn layer_paint(opacity: f32) -> u32 {
    (COLOR_SOURCE_LAYER << 29) | u32::from(pack_opacity(opacity))
}
