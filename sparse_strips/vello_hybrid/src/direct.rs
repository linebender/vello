// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Direct strip rendering without coarse rasterization or layer scheduling.

use crate::render::GpuStrip;
use crate::scene::{FastPathRect, FastStripCommand, FastStripsPath, RecordedRoot};
use crate::{Scene, TextureId};
use alloc::vec::Vec;
use vello_common::TextureId as CommonTextureId;
use vello_common::encode::{EncodedKind, EncodedPaint};
use vello_common::paint::{ImageSource, Paint};
use vello_common::strip_generator::StripStorage;
use vello_common::tile::Tile;

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

/// The final target direct strips will render into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DirectTarget {
    /// The user-provided surface.
    UserSurface,
    /// An atlas layer.
    AtlasLayer,
}

/// Origin of the current render target in scene coordinates.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RenderOrigin {
    pub(crate) x: u16,
    pub(crate) y: u16,
}

/// Specifies a run of strips inside [`DirectStrips`] that can be drawn with the same external
/// texture binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExternalTextureRun {
    pub(crate) texture_id: TextureId,
    /// Start index of the strip range for this run. The end is implicitly the start of the next
    /// run, or, for the last run, the total number of strips.
    pub(crate) strips_start: usize,
}

/// GPU strips generated from the scene's direct command stream.
#[derive(Debug, Default)]
pub(crate) struct DirectStrips {
    opaque: Vec<GpuStrip>,
    alpha: Vec<GpuStrip>,
    external_texture_runs: Vec<ExternalTextureRun>,
    last_alpha_external_texture_id: Option<TextureId>,
}

impl DirectStrips {
    #[allow(dead_code, reason = "Used by the WebGL backend")]
    pub(crate) fn from_scene(
        scene: &Scene,
        target: DirectTarget,
        paint_idxs: &[u32],
        encoded_paints: &[EncodedPaint],
    ) -> Self {
        Self::from_root(
            scene,
            scene.root(scene.root_id()),
            target,
            paint_idxs,
            encoded_paints,
        )
    }

    #[allow(
        dead_code,
        reason = "Kept for callers that already have a recorded root"
    )]
    pub(crate) fn from_root(
        scene: &Scene,
        root: &RecordedRoot,
        target: DirectTarget,
        paint_idxs: &[u32],
        encoded_paints: &[EncodedPaint],
    ) -> Self {
        let commands = root.direct_commands_without_layers();
        Self::from_commands(scene, &commands, target, paint_idxs, encoded_paints)
    }

    pub(crate) fn from_commands(
        scene: &Scene,
        commands: &[FastStripCommand],
        target: DirectTarget,
        paint_idxs: &[u32],
        encoded_paints: &[EncodedPaint],
    ) -> Self {
        let mut strips = Self::default();
        let strip_storage = scene.strip_storage.borrow();
        let mut depth = DepthCounter::default();
        let allow_opaque_split = target == DirectTarget::UserSurface;

        for cmd in commands {
            match cmd {
                FastStripCommand::Path(path) => {
                    let is_opaque = is_paint_opaque(&path.paint, encoded_paints);
                    let is_depth_opaque = is_opaque && allow_opaque_split;
                    let depth_index = depth.next(is_depth_opaque);
                    generate_gpu_strips_for_path(
                        path,
                        &strip_storage,
                        scene,
                        encoded_paints,
                        paint_idxs,
                        depth_index,
                        is_depth_opaque,
                        &mut strips,
                    );
                }
                FastStripCommand::Rect(rect) => {
                    let is_opaque = is_paint_opaque(&rect.paint, encoded_paints);
                    let is_depth_opaque = is_opaque && allow_opaque_split;
                    let depth_index = depth.next(is_depth_opaque);
                    pack_rectangle_into_gpu(
                        rect,
                        encoded_paints,
                        paint_idxs,
                        depth_index,
                        is_depth_opaque,
                        &mut strips,
                    );
                }
            }
        }

        strips.opaque.reverse();
        strips
    }

    pub(crate) fn opaque(&self) -> &[GpuStrip] {
        &self.opaque
    }

    pub(crate) fn alpha(&self) -> &[GpuStrip] {
        &self.alpha
    }

    pub(crate) fn external_texture_runs(&self) -> &[ExternalTextureRun] {
        &self.external_texture_runs
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.opaque.is_empty() && self.alpha.is_empty()
    }

    #[inline(always)]
    fn push_opaque(&mut self, gpu_strip: GpuStrip) {
        self.opaque.push(gpu_strip);
    }

    #[inline(always)]
    fn push_alpha(&mut self, gpu_strip: GpuStrip, external_texture_id: Option<CommonTextureId>) {
        if external_texture_id != self.last_alpha_external_texture_id {
            if let Some(texture_id) = external_texture_id {
                self.external_texture_runs.push(ExternalTextureRun {
                    strips_start: self.alpha.len(),
                    texture_id,
                });
            }
            self.last_alpha_external_texture_id = external_texture_id;
        }

        self.alpha.push(gpu_strip);
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

#[derive(Clone, Copy)]
struct ProcessedPaint {
    payload: u32,
    paint: u32,
    external_texture_id: Option<TextureId>,
}

/// Determine if a paint is fully opaque.
#[inline]
fn is_paint_opaque(paint: &Paint, encoded_paints: &[EncodedPaint]) -> bool {
    match paint {
        Paint::Solid(color) => color.is_opaque(),
        Paint::Indexed(indexed_paint) => {
            let paint_id = indexed_paint.index();
            match encoded_paints.get(paint_id) {
                Some(EncodedPaint::Image(img)) => {
                    !img.may_have_transparency
                        && img.sampler.alpha == 1.0
                        && img.tint.is_none_or(|t| t.color.components[3] >= 1.0)
                }
                Some(EncodedPaint::ExternalTexture(g)) => {
                    debug_assert!(
                        g.may_have_transparency,
                        "Front-to-back drawing of known-opaque external textures has not been implemented yet, so vello_hybrid always sets `may_have_transparency` to `true` for now."
                    );
                    false
                }
                Some(EncodedPaint::Gradient(g)) => !g.may_have_transparency,
                Some(EncodedPaint::BlurredRoundedRect(_)) => false,
                None => unreachable!("paint must be in encoded paints"),
            }
        }
    }
}

/// Process a paint and return the packed payload, paint and optional external texture id.
#[inline(always)]
fn process_paint(
    paint: &Paint,
    encoded_paints: &[EncodedPaint],
    (scene_strip_x, scene_strip_y): (u16, u16),
    paint_idxs: &[u32],
) -> ProcessedPaint {
    match paint {
        Paint::Solid(color) => {
            let rgba = color.as_premul_rgba8().to_u32();
            debug_assert!(
                rgba >= 0x1_00_00_00,
                "Color fields with 0 alpha are reserved for clipping"
            );
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
                Some(e) => process_encoded_paint(e, paint_idx, scene_strip_x, scene_strip_y),
                None => unimplemented!("unsupported paint type"),
            }
        }
    }
}

fn process_encoded_paint(
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
            _ => unimplemented!("unsupported image source"),
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
struct GpuStripBuilder {
    x: u16,
    y: u16,
    width: u16,
    dense_width_or_rect_height: u16,
    col_idx_or_rect_frac: u32,
}

impl GpuStripBuilder {
    /// Position at surface coordinates.
    fn at_surface(x: u16, y: u16, width: u16) -> Self {
        Self {
            x,
            y,
            width,
            dense_width_or_rect_height: 0,
            col_idx_or_rect_frac: 0,
        }
    }

    /// Add sparse strip parameters.
    fn with_sparse(mut self, dense_width: u16, col_idx: u32) -> Self {
        self.dense_width_or_rect_height = dense_width;
        self.col_idx_or_rect_frac = col_idx;
        self
    }

    /// Paint into strip.
    fn paint(self, payload: u32, paint: u32, depth_index: u32) -> GpuStrip {
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

fn generate_gpu_strips_for_path(
    path: &FastStripsPath,
    strip_storage: &StripStorage,
    scene: &Scene,
    encoded_paints: &[EncodedPaint],
    paint_idxs: &[u32],
    depth_index: u32,
    is_opaque: bool,
    strips_out: &mut DirectStrips,
) {
    let strips = &strip_storage.strips[path.strips.clone()];

    if strips.is_empty() {
        return;
    }

    for pair in strips.windows(2) {
        let strip = &pair[0];

        if strip.x >= scene.width {
            continue;
        }

        let next_strip = &pair[1];
        let col = strip.alpha_idx() / u32::from(Tile::HEIGHT);
        let next_col = next_strip.alpha_idx() / u32::from(Tile::HEIGHT);
        let strip_width = next_col.saturating_sub(col) as u16;
        let x0 = strip.x;
        let y = strip.y;

        // Alpha fill for the strip's coverage region.
        if strip_width > 0 {
            let processed = process_paint(&path.paint, encoded_paints, (x0, y), paint_idxs);
            strips_out.push_alpha(
                GpuStripBuilder::at_surface(x0, y, strip_width)
                    .with_sparse(strip_width, col)
                    .paint(processed.payload, processed.paint, depth_index),
                processed.external_texture_id,
            );
        }

        // Solid fill for the gap to the next strip.
        if next_strip.fill_gap() && strip.strip_y() == next_strip.strip_y() {
            let x1 = x0.saturating_add(strip_width);
            let x2 = next_strip.x.min(scene.width);
            if x2 > x1 {
                let processed = process_paint(&path.paint, encoded_paints, (x1, y), paint_idxs);
                let strip = GpuStripBuilder::at_surface(x1, y, x2 - x1).paint(
                    processed.payload,
                    processed.paint,
                    depth_index,
                );
                if is_opaque {
                    strips_out.push_opaque(strip);
                } else {
                    strips_out.push_alpha(strip, processed.external_texture_id);
                }
            }
        }
    }
}

fn pack_rectangle_into_gpu(
    rect: &FastPathRect,
    encoded_paints: &[EncodedPaint],
    paint_idxs: &[u32],
    depth_index: u32,
    is_opaque: bool,
    strips_out: &mut DirectStrips,
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
        let strip = make_gpu_rect(part, processed.payload, processed.paint, depth_index);
        if is_first && is_opaque && part.frac == 0 {
            strips_out.push_opaque(strip);
        } else {
            strips_out.push_alpha(strip, processed.external_texture_id);
        }
        is_first = false;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RectPart {
    x: u16,
    y: u16,
    width: u16,
    height: u16,
    frac: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SplitRect {
    main: RectPart,
    top: Option<RectPart>,
    bottom: Option<RectPart>,
    left: Option<RectPart>,
    right: Option<RectPart>,
}

fn split_rect(rect: &FastPathRect) -> SplitRect {
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

fn make_gpu_rect(part: RectPart, payload: u32, paint_packed: u32, depth_index: u32) -> GpuStrip {
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

fn pack_unorm4x8(v: [f32; 4]) -> u32 {
    let q = |f: f32| -> u8 { (f * 255.0 + 0.5) as u8 };
    u32::from(q(v[0]))
        | (u32::from(q(v[1])) << 8)
        | (u32::from(q(v[2])) << 16)
        | (u32::from(q(v[3])) << 24)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vello_common::color::palette::css::RED;

    fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> FastPathRect {
        FastPathRect {
            x0,
            y0,
            x1,
            y1,
            paint: RED.into(),
        }
    }

    #[test]
    fn split_small_rect_keeps_aa_fracs() {
        let split = split_rect(&rect(1.25, 2.5, 9.75, 10.0));
        assert_eq!(split.top, None);
        assert_eq!(split.bottom, None);
        assert_eq!(split.left, None);
        assert_eq!(split.right, None);
        assert_ne!(split.main.frac, 0);
    }

    #[test]
    fn split_large_aligned_rect_keeps_single_opaque_part() {
        let split = split_rect(&rect(0.0, 0.0, 64.0, 64.0));
        assert_eq!(split.main.frac, 0);
        assert_eq!(split.top, None);
        assert_eq!(split.bottom, None);
        assert_eq!(split.left, None);
        assert_eq!(split.right, None);
    }
}
