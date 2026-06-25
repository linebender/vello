// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared GPU strip packing helpers for `vello_hybrid`.

use crate::GpuStrip;
use crate::scene::FastPathRect;
use vello_common::TextureId;
use vello_common::encode::{EncodedKind, EncodedPaint};
use vello_common::paint::{ImageSource, Paint};

const COLOR_SOURCE_PAYLOAD: u32 = 0;

const PAINT_TYPE_SOLID: u32 = 0;
const PAINT_TYPE_IMAGE: u32 = 1;
const PAINT_TYPE_LINEAR_GRADIENT: u32 = 2;
const PAINT_TYPE_RADIAL_GRADIENT: u32 = 3;
const PAINT_TYPE_SWEEP_GRADIENT: u32 = 4;
const PAINT_TYPE_BLURRED_ROUNDED_RECT: u32 = 5;

/// Bit 31 of [`GpuStrip::paint_and_rect_flag`] signals that the strip
/// represents a full rectangle.
const RECT_STRIP_FLAG: u32 = 1 << 31;
/// The threshold of the rectangle size after which a rectangle should be split up
/// into multiple smaller ones.
const LARGE_RECT_SPLIT_THRESHOLD: u16 = 32;

#[derive(Clone, Copy)]
pub(crate) struct ProcessedPaint {
    pub(crate) payload: u32,
    pub(crate) paint: u32,
    pub(crate) external_texture_id: Option<TextureId>,
}

/// Process a paint and return the packed payload, paint and optional external texture id.
#[inline(always)]
pub(crate) fn process_paint(
    paint: &Paint,
    encoded_paints: &[EncodedPaint],
    (scene_strip_x, scene_strip_y): (u16, u16),
    paint_idxs: &[u32],
) -> ProcessedPaint {
    match paint {
        Paint::Solid(color) => {
            let rgba = color.as_premul_rgba8().to_u32();
            let paint_packed = (COLOR_SOURCE_PAYLOAD << 30) | (PAINT_TYPE_SOLID << 27);
            ProcessedPaint {
                payload: rgba,
                paint: paint_packed,
                external_texture_id: None,
            }
        }
        Paint::Indexed(indexed_paint) => {
            let paint_id = indexed_paint.index();
            let paint_idx = paint_idxs.get(paint_id).copied().unwrap();

            match encoded_paints.get(paint_id) {
                Some(encoded_paint) => {
                    process_encoded_paint(encoded_paint, paint_idx, scene_strip_x, scene_strip_y)
                }
                None => unimplemented!("Unsupported paint type"),
            }
        }
    }
}

pub(crate) fn process_encoded_paint(
    encoded_paint: &EncodedPaint,
    paint_idx: u32,
    scene_strip_x: u16,
    scene_strip_y: u16,
) -> ProcessedPaint {
    match encoded_paint {
        EncodedPaint::Image(encoded_image) => match &encoded_image.source {
            ImageSource::OpaqueId { .. } => {
                let paint_packed = (COLOR_SOURCE_PAYLOAD << 29)
                    | (PAINT_TYPE_IMAGE << 26)
                    | (paint_idx & 0x03FF_FFFF);
                let scene_strip_xy = ((scene_strip_y as u32) << 16) | (scene_strip_x as u32);
                ProcessedPaint {
                    payload: scene_strip_xy,
                    paint: paint_packed,
                    external_texture_id: None,
                }
            }
            _ => unimplemented!("Unsupported image source"),
        },
        EncodedPaint::ExternalTexture(texture) => {
            let paint_packed =
                (COLOR_SOURCE_PAYLOAD << 29) | (PAINT_TYPE_IMAGE << 26) | (paint_idx & 0x03FF_FFFF);
            let scene_strip_xy = ((scene_strip_y as u32) << 16) | (scene_strip_x as u32);
            ProcessedPaint {
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
            ProcessedPaint {
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
            ProcessedPaint {
                payload: scene_strip_xy,
                paint: paint_packed,
                external_texture_id: None,
            }
        }
    }
}

/// Helper for more semantically constructing `GpuStrip`s.
pub(crate) struct GpuStripBuilder {
    x: u16,
    y: u16,
    width: u16,
    dense_width_or_rect_height: u16,
    col_idx_or_rect_frac: u32,
}

impl GpuStripBuilder {
    /// Position at surface coordinates.
    pub(crate) fn at_surface(x: u16, y: u16, width: u16) -> Self {
        Self {
            x,
            y,
            width,
            dense_width_or_rect_height: 0,
            col_idx_or_rect_frac: 0,
        }
    }

    /// Add sparse strip parameters.
    pub(crate) fn with_sparse(mut self, dense_width: u16, col_idx: u32) -> Self {
        self.dense_width_or_rect_height = dense_width;
        self.col_idx_or_rect_frac = col_idx;
        self
    }

    /// Paint into strip.
    pub(crate) fn paint(self, payload: u32, paint: u32, depth_index: u32) -> GpuStrip {
        GpuStrip {
            x: self.x,
            y: self.y,
            width: self.width,
            dense_width_or_rect_height: self.dense_width_or_rect_height,
            col_idx_or_rect_frac: self.col_idx_or_rect_frac,
            payload,
            paint_and_rect_flag: paint,
            depth_index,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RectPart {
    pub(crate) x: u16,
    pub(crate) y: u16,
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) frac: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SplitRect {
    pub(crate) main: RectPart,
    pub(crate) top: Option<RectPart>,
    pub(crate) bottom: Option<RectPart>,
    pub(crate) left: Option<RectPart>,
    pub(crate) right: Option<RectPart>,
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "recorded rect coordinates are clipped to the u16 viewport domain before packing"
)]
pub(crate) fn split_rect(rect: &FastPathRect) -> SplitRect {
    let sx0 = rect.x0.floor();
    let sy0 = rect.y0.floor();
    let sx1 = rect.x1.ceil();
    let sy1 = rect.y1.ceil();

    let x = sx0 as u16;
    let y = sy0 as u16;
    let width = (sx1 - sx0) as u16;
    let height = (sy1 - sy0) as u16;

    let left_frac = rect.x0 - sx0;
    let top_frac = rect.y0 - sy0;
    let right_frac = sx1 - rect.x1;
    let bottom_frac = sy1 - rect.y1;

    if rect.x1 - rect.x0 < f32::from(LARGE_RECT_SPLIT_THRESHOLD)
        || rect.y1 - rect.y0 < f32::from(LARGE_RECT_SPLIT_THRESHOLD)
    {
        return SplitRect {
            main: RectPart {
                x,
                y,
                width,
                height,
                frac: pack_unorm4x8([left_frac, top_frac, right_frac, bottom_frac]),
            },
            top: None,
            bottom: None,
            left: None,
            right: None,
        };
    }

    let has_left_aa = left_frac > 0.0;
    let has_top_aa = top_frac > 0.0;
    let has_right_aa = right_frac > 0.0;
    let has_bottom_aa = bottom_frac > 0.0;
    let has_top_strip = has_top_aa || has_left_aa || has_right_aa;
    let has_bottom_strip = has_bottom_aa || has_left_aa || has_right_aa;
    let left_inset = u16::from(has_left_aa);
    let right_inset = u16::from(has_right_aa);
    let top_inset = u16::from(has_top_strip);
    let bottom_inset = u16::from(has_bottom_strip);
    let inner_x = x + left_inset;
    let inner_y = y + top_inset;
    let inner_width = width - left_inset - right_inset;
    let inner_height = height - top_inset - bottom_inset;

    SplitRect {
        main: RectPart {
            x: inner_x,
            y: inner_y,
            width: inner_width,
            height: inner_height,
            frac: 0,
        },
        top: has_top_strip.then_some(RectPart {
            x,
            y,
            width,
            height: 1,
            frac: pack_unorm4x8([left_frac, top_frac, right_frac, 0.0]),
        }),
        bottom: has_bottom_strip.then_some(RectPart {
            x,
            y: y + height - 1,
            width,
            height: 1,
            frac: pack_unorm4x8([left_frac, 0.0, right_frac, bottom_frac]),
        }),
        left: has_left_aa.then_some(RectPart {
            x,
            y: inner_y,
            width: 1,
            height: inner_height,
            frac: pack_unorm4x8([left_frac, 0.0, 0.0, 0.0]),
        }),
        right: has_right_aa.then_some(RectPart {
            x: x + width - 1,
            y: inner_y,
            width: 1,
            height: inner_height,
            frac: pack_unorm4x8([0.0, 0.0, right_frac, 0.0]),
        }),
    }
}

pub(crate) fn make_gpu_rect(
    part: RectPart,
    payload: u32,
    paint_packed: u32,
    depth_index: u32,
) -> GpuStrip {
    GpuStrip {
        x: part.x,
        y: part.y,
        width: part.width,
        dense_width_or_rect_height: part.height,
        col_idx_or_rect_frac: part.frac,
        payload,
        paint_and_rect_flag: paint_packed | RECT_STRIP_FLAG,
        depth_index,
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "normalized color channels are packed into u8 lanes"
)]
fn pack_unorm4x8(v: [f32; 4]) -> u32 {
    let q = |f: f32| -> u8 { (f * 255.0 + 0.5) as u8 };
    u32::from(q(v[0]))
        | (u32::from(q(v[1])) << 8)
        | (u32::from(q(v[2])) << 16)
        | (u32::from(q(v[3])) << 24)
}

#[cfg(test)]
mod tests {
    use super::{RectPart, SplitRect, pack_unorm4x8, split_rect};
    use crate::scene::FastPathRect;
    use vello_common::paint::{Color, Paint};

    fn solid_rect(x0: f32, y0: f32, x1: f32, y1: f32) -> FastPathRect {
        FastPathRect {
            x0,
            y0,
            x1,
            y1,
            paint: Paint::from(Color::from_rgba8(255, 0, 0, 255)),
        }
    }

    fn part(x: u16, y: u16, width: u16, height: u16, frac: [f32; 4]) -> RectPart {
        RectPart {
            x,
            y,
            width,
            height,
            frac: pack_unorm4x8(frac),
        }
    }

    #[test]
    fn splitter_keeps_small_rect_whole() {
        let rect = solid_rect(10.25, 20.5, 25.75, 35.25);
        let split = split_rect(&rect);

        assert_eq!(
            split,
            SplitRect {
                main: part(10, 20, 16, 16, [0.25, 0.5, 0.25, 0.75]),
                top: None,
                bottom: None,
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_splits_large_rect_into_five_parts() {
        let rect = solid_rect(10.25, 20.5, 42.75, 52.75);
        let split = split_rect(&rect);

        assert_eq!(
            split,
            SplitRect {
                main: part(11, 21, 31, 31, [0.0, 0.0, 0.0, 0.0]),
                top: Some(part(10, 20, 33, 1, [0.25, 0.5, 0.25, 0.0])),
                bottom: Some(part(10, 52, 33, 1, [0.25, 0.0, 0.25, 0.25])),
                left: Some(part(10, 21, 1, 31, [0.25, 0.0, 0.0, 0.0])),
                right: Some(part(42, 21, 1, 31, [0.0, 0.0, 0.25, 0.0])),
            }
        );
    }

    #[test]
    fn splitter_omits_unneeded_edge_parts() {
        let rect = solid_rect(10.0, 20.5, 42.0, 53.0);
        let split = split_rect(&rect);

        assert_eq!(
            split,
            SplitRect {
                main: part(10, 21, 32, 32, [0.0, 0.0, 0.0, 0.0]),
                top: Some(part(10, 20, 32, 1, [0.0, 0.5, 0.0, 0.0])),
                bottom: None,
                left: None,
                right: None,
            }
        );
    }
}
