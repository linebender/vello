// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Draw construction for scheduled strip render passes.

use super::{ExternalTextureRun, LayerTextureRegion};
use crate::GpuStrip;
use crate::rect::{RectPart, make_gpu_rect, split_rect};
use crate::scene::RecordedDraw;
use alloc::vec::Vec;
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
    pub(super) opaque: Vec<GpuStrip>,
    pub(super) alpha: Vec<GpuStrip>,
    pub(super) external_texture_runs: Vec<ExternalTextureRun>,
}

impl Draw {
    pub(super) fn append_opaque(&mut self, other: &Self) {
        self.opaque.extend_from_slice(&other.opaque);
    }

    pub(super) fn append(&mut self, other: &Self) {
        self.opaque.extend_from_slice(&other.opaque);

        let alpha_offset = self.alpha.len();
        let had_external_texture_runs = !self.external_texture_runs.is_empty();
        for (idx, run) in other.external_texture_runs.iter().enumerate() {
            let strips_start = if !had_external_texture_runs && idx == 0 {
                0
            } else {
                alpha_offset + run.strips_start
            };
            if strips_start == self.alpha.len()
                && self
                    .external_texture_runs
                    .last()
                    .is_some_and(|last| last.texture_id == run.texture_id)
            {
                continue;
            }
            self.external_texture_runs.push(ExternalTextureRun {
                texture_id: run.texture_id,
                strips_start,
            });
        }
        self.alpha.extend_from_slice(&other.alpha);
    }

    #[inline(always)]
    fn push_opaque(&mut self, gpu_strip: GpuStrip) {
        self.opaque.push(gpu_strip);
    }

    #[inline(always)]
    fn push_alpha(&mut self, gpu_strip: GpuStrip, external_texture_id: Option<TextureId>) {
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
        self.opaque.is_empty() && self.alpha.is_empty()
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct LayerSample {
    pub(super) source: LayerTextureRegion,
    pub(super) bbox: RectU16,
    pub(super) source_origin: (u16, u16),
}

#[derive(Debug)]
pub(super) struct DrawBuilder {
    draw: Draw,
    depth: DepthCounter,
    allow_opaque_pass: bool,
    geometry_offset: (i32, i32),
    draw_bounds: RectU16,
}

impl DrawBuilder {
    pub(super) fn new(
        allow_opaque_pass: bool,
        geometry_offset: (i32, i32),
        draw_bounds: RectU16,
    ) -> Self {
        Self {
            draw: Draw::default(),
            depth: DepthCounter::default(),
            allow_opaque_pass,
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

        let is_opaque = self.allow_opaque_pass && is_paint_opaque(&paint, encoded_paints);
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
                    self.draw.push_alpha(
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
                    if is_opaque {
                        self.draw.push_opaque(strip);
                    } else {
                        self.draw.push_alpha(strip, processed.external_texture_id);
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
        let Some(rect) = clipped_fast_rect(rect, self.draw_bounds) else {
            return;
        };
        let is_opaque = self.allow_opaque_pass && is_paint_opaque(paint, encoded_paints);
        let depth_index = self.depth.next(is_opaque);
        pack_rectangle_into_gpu(
            &rect,
            paint,
            encoded_paints,
            paint_idxs,
            depth_index,
            is_opaque,
            self.geometry_offset,
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
                self.draw.push_alpha(
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
                self.draw.push_alpha(
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

    pub(super) fn take_draw(&mut self) -> Draw {
        let mut draw = core::mem::take(&mut self.draw);
        draw.opaque.reverse();
        draw
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
    rect: &Rect,
    paint: &Paint,
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
        let processed = pack_paint(paint, encoded_paints, (part.x, part.y), paint_idxs);
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

fn clipped_fast_rect(rect: &Rect, bbox: RectU16) -> Option<Rect> {
    let x0 = rect.x0.max(f64::from(bbox.x0));
    let y0 = rect.y0.max(f64::from(bbox.y0));
    let x1 = rect.x1.min(f64::from(bbox.x1));
    let y1 = rect.y1.min(f64::from(bbox.y1));

    (x0 < x1 && y0 < y1).then(|| Rect::new(x0, y0, x1, y1))
}

fn tile_bounds(bounds: RectU16) -> RectU16 {
    RectU16::new(
        bounds.x0 / Tile::WIDTH,
        bounds.y0 / Tile::HEIGHT,
        bounds.x1.div_ceil(Tile::WIDTH),
        bounds.y1.div_ceil(Tile::HEIGHT),
    )
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
    debug_assert!(
        (0..=i32::from(u16::MAX)).contains(&coord),
        "offset coordinate must fit into u16"
    );
    u16::try_from(coord).expect("offset coordinate must fit into u16")
}

fn pack_u16_pair(x: u32, y: u32) -> u32 {
    debug_assert!(x <= u32::from(u16::MAX), "x payload must fit into u16");
    debug_assert!(y <= u32::from(u16::MAX), "y payload must fit into u16");
    (x & 0xffff) | ((y & 0xffff) << 16)
}

fn layer_sample_payload(sample: LayerSample, x: u16, y: u16) -> u32 {
    let source = sample.source;
    let source_x =
        source.x + u32::from(sample.source_origin.0) + u32::from(x - source.scene_bbox.x0);
    let source_y =
        source.y + u32::from(sample.source_origin.1) + u32::from(y - source.scene_bbox.y0);
    pack_u16_pair(source_x, source_y)
}

fn layer_paint(opacity: f32) -> u32 {
    (COLOR_SOURCE_LAYER << 29) | u32::from(opacity_to_u8(opacity))
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "opacity is clamped to the normalized u8 range before packing"
)]
fn opacity_to_u8(opacity: f32) -> u8 {
    (opacity.clamp(0.0, 1.0) * 255.0).round() as u8
}
