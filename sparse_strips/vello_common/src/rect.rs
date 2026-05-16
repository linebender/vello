// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Fast pixel-aligned rectangle rendering directly into strips.

use crate::kurbo::Rect;
#[cfg(not(feature = "std"))]
use crate::kurbo::common::FloatFuncs as _;
use crate::strip::{Strip, StripExt};
use crate::tile::{LargeSize, MAX_TILE_DIMENSION, MediumSize, SmallSize, TileSize};
use alloc::vec::Vec;
use fearless_simd::*;

/// Tile-size policy needed by fast rectangle rendering.
pub(crate) trait RectExt: StripExt {
    /// Append one tile's x-only alpha mask.
    fn append_x_alpha_tile<S: Simd>(simd: S, x_cov: &[f32], alpha_buf: &mut Vec<u8>) {
        debug_assert_eq!(x_cov.len(), usize::from(Self::WIDTH));

        let scale = Self::WindingVector::<S>::splat(simd, 255.0);
        let rounding = Self::WindingVector::<S>::splat(simd, 0.5);
        let mut columns = [Self::WindingVector::<S>::splat(simd, 0.0); MAX_TILE_DIMENSION];

        for (column, xc) in columns.iter_mut().zip(x_cov.iter().copied()) {
            *column = Self::WindingVector::<S>::splat(simd, xc).mul_add(scale, rounding);
        }

        Self::append_alpha_tile(simd, &columns[..usize::from(Self::WIDTH)], alpha_buf);
    }

    /// Append one tile's alpha mask, combining x and y coverage.
    fn append_combined_tile_alpha<S: Simd>(
        simd: S,
        x_cov: &[f32],
        y_cov: &[f32],
        alpha_buf: &mut Vec<u8>,
    ) {
        debug_assert_eq!(x_cov.len(), usize::from(Self::WIDTH));
        debug_assert_eq!(y_cov.len(), usize::from(Self::HEIGHT));

        let y_cov = Self::WindingVector::<S>::from_slice(simd, y_cov);
        let scale = Self::WindingVector::<S>::splat(simd, 255.0);
        let rounding = Self::WindingVector::<S>::splat(simd, 0.5);
        let mut columns = [Self::WindingVector::<S>::splat(simd, 0.0); MAX_TILE_DIMENSION];

        for (column, xc) in columns.iter_mut().zip(x_cov.iter().copied()) {
            *column = y_cov.mul_add(scale * Self::WindingVector::<S>::splat(simd, xc), rounding);
        }

        Self::append_alpha_tile(simd, &columns[..usize::from(Self::WIDTH)], alpha_buf);
    }
}

impl RectExt for SmallSize {}
impl RectExt for MediumSize {}
impl RectExt for LargeSize {}

/// Render a pixel-aligned rectangle directly into strips.
///
/// This bypasses the full path processing pipeline (flatten → tiles → strips)
/// by directly creating strip coverage data for the rectangle.
///
/// The rect bounds should already be clamped to the viewport.
pub fn render<S: TileSize>(
    level: Level,
    rect: Rect,
    strip_buf: &mut Vec<Strip<S>>,
    alpha_buf: &mut Vec<u8>,
) {
    dispatch!(level, simd => render_impl::<_, S>(simd, rect, strip_buf, alpha_buf));
}

/// Generates strip data for an axis-aligned rectangle.
///
/// # Strip layout strategy
///
/// Tile rows are classified into two kinds:
///
/// - **Edge rows** (top/bottom of rect): the rect boundary crosses partway
///   through the tile vertically, so individual pixels need per-cell alpha.
///   We emit a *single wide strip* spanning all tile columns, with alpha =
///   `x_alpha` * `y_alpha` (so the intersection of the alpha mask in each direction).
///
/// - **Interior rows**: every pixel in the tile has full vertical coverage,
///   so we only need to handle the left and right partial-column edges.
///   We emit a **left edge strip** (with its x-alpha mask) and, when the rect
///   spans more than one tile column, a **right edge strip** with `fill_gap =
///   true` so the renderer fills solid 0xFF between them.
///
/// The horizontal coverage for the left/right edge tiles is y-independent, so it
/// is precomputed once and reused across all interior rows.
fn render_impl<S: Simd, TS: TileSize>(
    s: S,
    rect: Rect,
    strip_buf: &mut Vec<Strip<TS>>,
    alpha_buf: &mut Vec<u8>,
) {
    if rect.is_zero_area() {
        return;
    }

    let rect_x0 = rect.x0 as f32;
    let rect_y0 = rect.y0 as f32;
    let rect_x1 = rect.x1 as f32;
    let rect_y1 = rect.y1 as f32;

    // Integer pixel bounds.
    let px_x0 = rect_x0.floor() as u16;
    let px_y0 = rect_y0.floor() as u16;
    let px_y1 = rect_y1.ceil() as u16;

    let left_tile_x = (px_x0 / TS::WIDTH) * TS::WIDTH;
    // Inclusive, so don't use `ceil` here but just `rect_x1` directly.
    let right_tile_x = (rect_x1 as u16 / TS::WIDTH) * TS::WIDTH;

    let y0 = (px_y0 / TS::HEIGHT) * TS::HEIGHT;
    // Note: y1 is exclusive, but the last tile cannot be rounded up further if
    // the viewport height reaches u16::MAX.
    let y1 = (px_y1.saturating_add(TS::HEIGHT - 1) / TS::HEIGHT) * TS::HEIGHT;
    // Include one tile past the right edge so the right-edge tile column is
    // covered by the edge-row wide-strip loop.
    let x_end = right_tile_x.saturating_add(TS::WIDTH);

    if x_end <= left_tile_x || y1 <= y0 {
        return;
    }

    let tile_start_y = y0 / TS::HEIGHT;
    let tile_end_y = y1 / TS::HEIGHT;

    // A right strip is only needed when the rect spans more than one tile column.
    let needs_right_strip = right_tile_x > left_tile_x;

    let left_x_cov = coverage(usize::from(TS::WIDTH), left_tile_x, rect_x0, rect_x1);
    let right_x_cov = coverage(usize::from(TS::WIDTH), right_tile_x, rect_x0, rect_x1);

    for tile_y in tile_start_y..tile_end_y {
        let strip_y = tile_y * TS::HEIGHT;
        let strip_y_f = strip_y as f32;
        let strip_y_end_f = strip_y as f32 + TS::HEIGHT as f32;

        // A row is an "edge" if the rect's top or bottom boundary falls
        // *inside* it (i.e. partial vertical coverage).
        let is_top_edge = strip_y_f < rect_y0 && rect_y0 < strip_y_end_f;
        let is_bottom_edge = strip_y_f < rect_y1 && rect_y1 < strip_y_end_f;

        if is_top_edge || is_bottom_edge {
            let alpha_start = alpha_buf.len() as u32;

            let y_cov = coverage(usize::from(TS::HEIGHT), strip_y, rect_y0, rect_y1);
            let mut col = left_tile_x;
            // TODO: Can this result in an infinite loop in case x_end == u16::MAX?
            while col + TS::WIDTH <= x_end {
                // TODO: We could optimize this so this is only computed for the left-most and right-most
                // tile of the edge, all intermediate tiles have full horizontal coverage.
                let x_cov = coverage(usize::from(TS::WIDTH), col, rect_x0, rect_x1);
                TS::append_combined_tile_alpha(
                    s,
                    &x_cov[..usize::from(TS::WIDTH)],
                    &y_cov[..usize::from(TS::HEIGHT)],
                    alpha_buf,
                );
                col += TS::WIDTH;
            }

            strip_buf.push(Strip::new(left_tile_x, strip_y, alpha_start, false));
        } else {
            let alpha_start = alpha_buf.len() as u32;
            TS::append_x_alpha_tile(s, &left_x_cov[..usize::from(TS::WIDTH)], alpha_buf);
            strip_buf.push(Strip::new(left_tile_x, strip_y, alpha_start, false));

            if needs_right_strip {
                // `fill_gap = true` tells the renderer to fill solid 0xFF
                // between the previous strip's end and this strip's start.
                let alpha_start = alpha_buf.len() as u32;
                TS::append_x_alpha_tile(s, &right_x_cov[..usize::from(TS::WIDTH)], alpha_buf);
                strip_buf.push(Strip::new(right_tile_x, strip_y, alpha_start, true));
            }
        }
    }

    // Sentinel strip: marks the end of the strip list for this shape.
    let last_strip_y = (tile_end_y - 1) * TS::HEIGHT;
    strip_buf.push(Strip::new(
        u16::MAX,
        last_strip_y,
        alpha_buf.len() as u32,
        false,
    ));
}

/// Compute fractional pixel coverage for consecutive pixels starting at `start`.
#[inline(always)]
fn coverage(len: usize, start: u16, rect_lo: f32, rect_hi: f32) -> [f32; MAX_TILE_DIMENSION] {
    debug_assert!(len <= MAX_TILE_DIMENSION);
    let mut cov = [0.0_f32; MAX_TILE_DIMENSION];

    #[allow(clippy::needless_range_loop, reason = "better clarity")]
    for i in 0..len {
        let px = (start as usize + i) as f32;
        cov[i] = (rect_hi.min(px + 1.0) - rect_lo.max(px)).clamp(0.0, 1.0);
    }
    cov
}
