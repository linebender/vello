// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Fast pixel-aligned rectangle rendering directly into strips.

use crate::kurbo::Rect;
#[cfg(not(feature = "std"))]
use crate::kurbo::common::FloatFuncs as _;
use crate::strip::Strip;
use crate::tile::Tile;
use alloc::vec::Vec;
use fearless_simd::*;

/// Render a pixel-aligned rectangle directly into strips.
///
/// This bypasses the full path processing pipeline (flatten → tiles → strips)
/// by directly creating strip coverage data for the rectangle.
///
/// The rect bounds should already be clamped to the viewport.
pub fn render(level: Level, rect: Rect, strip_buf: &mut Vec<Strip>, alpha_buf: &mut Vec<u8>) {
    dispatch!(level, simd => render_impl(simd, rect, strip_buf, alpha_buf));
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
/// The x-alpha masks for the left/right edge tiles are y-independent, so they
/// are precomputed once and reused across all interior rows.
#[inline(always)]
fn render_impl<S: Simd>(s: S, rect: Rect, strip_buf: &mut Vec<Strip>, alpha_buf: &mut Vec<u8>) {
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

    let left_tile_x = (px_x0 / Tile::WIDTH) * Tile::WIDTH;
    // Inclusive, so don't use `ceil` here but just `rect_x1` directly.
    let right_tile_x = (rect_x1 as u16 / Tile::WIDTH) * Tile::WIDTH;

    let y0 = u32::from((px_y0 / Tile::HEIGHT) * Tile::HEIGHT);
    let y1 = (u32::from(px_y1) + u32::from(Tile::HEIGHT - 1)) / u32::from(Tile::HEIGHT)
        * u32::from(Tile::HEIGHT);
    // Include one tile past the right edge so the right-edge tile column is
    // covered by the edge-row wide-strip loop.
    let x_end = u32::from(right_tile_x) + u32::from(Tile::WIDTH);

    if x_end <= u32::from(left_tile_x) || y1 <= y0 {
        return;
    }

    let tile_start_y = y0 / u32::from(Tile::HEIGHT);
    let tile_end_y = y1 / u32::from(Tile::HEIGHT);

    // A right strip is only needed when the rect spans more than one tile column.
    let needs_right_strip = right_tile_x > left_tile_x;

    let left_x_cov = coverage(left_tile_x, rect_x0, rect_x1);
    let right_x_cov = coverage(right_tile_x, rect_x0, rect_x1);
    let left_x_mask = alpha_mask_from_x_coverage(s, &left_x_cov);
    let right_x_mask = alpha_mask_from_x_coverage(s, &right_x_cov);

    for tile_y in tile_start_y..tile_end_y {
        let strip_y = tile_y * u32::from(Tile::HEIGHT);
        let strip_y = strip_y as u16;
        let strip_y_f = strip_y as f32;
        let strip_y_end_f = strip_y as f32 + Tile::HEIGHT as f32;

        // A row is an "edge" if the rect's top or bottom boundary falls
        // *inside* it (i.e. partial vertical coverage).
        let is_top_edge = strip_y_f < rect_y0 && rect_y0 < strip_y_end_f;
        let is_bottom_edge = strip_y_f < rect_y1 && rect_y1 < strip_y_end_f;

        if is_top_edge || is_bottom_edge {
            let alpha_start = alpha_buf.len() as u32;

            let y_cov = coverage(strip_y, rect_y0, rect_y1);
            let mut col = u32::from(left_tile_x);
            while col + u32::from(Tile::WIDTH) <= x_end {
                // TODO: We could optimize this so this is only computed for the left-most and right-most
                // tile of the edge, all intermediate tiles have full horizontal coverage.
                let x_cov = coverage(col as u16, rect_x0, rect_x1);
                let combined = combined_tile_alpha(s, &x_cov, &y_cov);
                alpha_buf.extend_from_slice(combined.as_slice());
                col += u32::from(Tile::WIDTH);
            }

            strip_buf.push(Strip::new(left_tile_x, strip_y, alpha_start, false));
        } else {
            let alpha_start = alpha_buf.len() as u32;
            alpha_buf.extend_from_slice(left_x_mask.as_slice());
            strip_buf.push(Strip::new(left_tile_x, strip_y, alpha_start, false));

            if needs_right_strip {
                // `fill_gap = true` tells the renderer to fill solid 0xFF
                // between the previous strip's end and this strip's start.
                let alpha_start = alpha_buf.len() as u32;
                alpha_buf.extend_from_slice(right_x_mask.as_slice());
                strip_buf.push(Strip::new(right_tile_x, strip_y, alpha_start, true));
            }
        }
    }

    // Sentinel strip: marks the end of the strip list for this shape.
    let last_strip_y = ((tile_end_y - 1) * u32::from(Tile::HEIGHT)) as u16;
    strip_buf.push(Strip::sentinel(last_strip_y, alpha_buf.len() as u32));
}

/// Compute fractional pixel coverage for `N` consecutive pixels starting at `start`.
#[inline(always)]
fn coverage<const N: usize>(start: u16, rect_lo: f32, rect_hi: f32) -> [f32; N] {
    let mut cov = [0.0_f32; N];

    #[allow(clippy::needless_range_loop, reason = "better clarity")]
    for i in 0..N {
        let px = (start as usize + i) as f32;
        cov[i] = pixel_coverage(px, rect_lo, rect_hi);
    }
    cov
}

/// How much of the 1-pixel span starting at `px` is covered by `[rect_lo, rect_hi]`, in `[0, 1]`.
#[inline(always)]
pub fn pixel_coverage(px: f32, rect_lo: f32, rect_hi: f32) -> f32 {
    (rect_hi.min(px + 1.0) - rect_lo.max(px)).clamp(0.0, 1.0)
}

/// Quantize a `[0, 1]` single-axis coverage value to a byte.
///
/// This is the quantization the strip renderer applies to axis-aligned rectangle coverage. The
/// GPU rect quad in `vello_hybrid` derives its per-pixel alpha from bytes produced by this
/// function and [`corner_coverage_u8`], which is what keeps the two rasterizers byte-identical.
#[inline(always)]
pub fn coverage_to_u8(cov: f32) -> u8 {
    (cov * 255.0 + 0.5) as u8
}

/// The alpha byte of a pixel covered by `cov_x * cov_y` of its area (a rectangle corner pixel),
/// quantized from the exact product — the same value [`render`] writes for that pixel.
#[inline(always)]
pub fn corner_coverage_u8(cov_x: f32, cov_y: f32) -> u8 {
    (cov_x * cov_y * 255.0 + 0.5) as u8
}

/// `round(x * y / 255)` computed exactly in integer arithmetic.
///
/// This approximates [`corner_coverage_u8`] from the two already-quantized axis bytes; the result
/// is always within 1 of it (quantizing each axis first loses strictly less than half an alpha
/// step per axis). The GPU rect quad evaluates this expression in its shader and applies a
/// precomputed 2-bit correction to land exactly on the [`corner_coverage_u8`] value.
#[inline(always)]
pub fn combine_coverage_u8(x: u8, y: u8) -> u8 {
    let p = u32::from(x) * u32::from(y) + 128;
    ((p + (p >> 8)) >> 8) as u8
}

/// Build an alpha mask for the 4x4 tile from the given horizontal coverages,
/// splatting them across the other dimension.
#[inline(always)]
fn alpha_mask_from_x_coverage<S: Simd>(s: S, cov: &[f32; Tile::WIDTH as usize]) -> u8x16<S> {
    let mut buf = [0_u8; 16];

    #[allow(clippy::needless_range_loop, reason = "better clarity")]
    for col in 0..Tile::WIDTH as usize {
        let alpha = coverage_to_u8(cov[col]);
        let base = col * Tile::HEIGHT as usize;
        buf[base..base + Tile::HEIGHT as usize].fill(alpha);
    }

    u8x16::from_slice(s, &buf)
}

/// Compute the alphas for a single 4x4 tile, taking horizontal as well as vertical coverage
/// of the rectangle into account.
#[inline(always)]
fn combined_tile_alpha<S: Simd>(
    s: S,
    x_cov: &[f32; Tile::WIDTH as usize],
    y_cov: &[f32; Tile::HEIGHT as usize],
) -> u8x16<S> {
    let mut buf = [0_u8; 16];
    for (col, xc) in x_cov.iter().copied().enumerate() {
        for (row, yc) in y_cov.iter().copied().enumerate() {
            buf[col * Tile::HEIGHT as usize + row] = corner_coverage_u8(xc, yc);
        }
    }

    u8x16::from_slice(s, &buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn render_edge_row_at_u16_right_edge() {
        let mut strips = Vec::new();
        let mut alphas = Vec::new();
        let rect = Rect::new(f64::from(u16::MAX - 3), 0.5, f64::from(u16::MAX), 3.5);

        render(Level::baseline(), rect, &mut strips, &mut alphas);

        assert_eq!(strips.len(), 2);
        assert_eq!(strips[0].x, u16::MAX - 3);
        assert_eq!(strips[0].alpha_idx(), 0);
        assert_eq!(
            alphas.len(),
            usize::from(Tile::WIDTH) * usize::from(Tile::HEIGHT)
        );
        assert!(strips[1].is_sentinel());
    }

    #[test]
    fn render_edge_row_at_u16_bottom_edge() {
        let mut strips = Vec::new();
        let mut alphas = Vec::new();
        let rect = Rect::new(0.5, f64::from(u16::MAX - 3), 3.5, f64::from(u16::MAX));

        render(Level::baseline(), rect, &mut strips, &mut alphas);

        assert_eq!(strips.len(), 2);
        assert_eq!(strips[0].y, u16::MAX - 3);
        assert_eq!(strips[0].alpha_idx(), 0);
        assert_eq!(
            alphas.len(),
            usize::from(Tile::WIDTH) * usize::from(Tile::HEIGHT)
        );
        assert!(strips[1].is_sentinel());
    }
}
