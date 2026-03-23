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

/// Generates strip data for a pixel-aligned rectangle.
///
/// # Strip layout strategy
///
/// Tile rows are classified into two kinds:
///
/// - **Edge rows** (top/bottom of rect): the rect boundary crosses partway
///   through the tile vertically, so individual pixels need per-cell alpha.
///   We emit a *single wide strip* spanning all tile columns, with alpha =
///   `x_mask & y_mask` (each is 0x00 or 0xFF, so AND gives the intersection).
///
/// - **Interior rows**: every pixel in the tile has full vertical coverage,
///   so we only need to handle the left and right partial-column edges.
///   We emit a **left edge strip** (with its x-alpha mask) and, when the rect
///   spans more than one tile column, a **right edge strip** with `fill_gap =
///   true` so the renderer fills solid 0xFF between them.
///
/// The x-alpha masks for the left/right edge tiles are y-independent, so they
/// are precomputed once and reused across all interior rows.
fn render_impl<S: Simd>(s: S, rect: Rect, strip_buf: &mut Vec<Strip>, alpha_buf: &mut Vec<u8>) {
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

    let y0 = (px_y0 / Tile::HEIGHT) * Tile::HEIGHT;
    // Note: y1 is exclusive, but it's gonna break for the very last tile if we have a height of u16::MAX.
    let y1 = (px_y1.saturating_add(Tile::HEIGHT - 1) / Tile::HEIGHT) * Tile::HEIGHT;
    // Include one tile past the right edge so the right-edge tile column is
    // covered by the edge-row wide-strip loop.
    let x_end = right_tile_x.saturating_add(Tile::WIDTH);

    if x_end <= left_tile_x || y1 <= y0 {
        return;
    }

    let tile_start_y = y0 / Tile::HEIGHT;
    let tile_end_y = y1 / Tile::HEIGHT;

    // A right strip is only needed when the rect spans more than one tile column.
    let needs_right_strip = right_tile_x > left_tile_x;

    let left_x_cov = coverage(left_tile_x, rect_x0, rect_x1);
    let right_x_cov = coverage(right_tile_x, rect_x0, rect_x1);
    let left_x_mask = alpha_mask_from_x_coverage(s, &left_x_cov);
    let right_x_mask = alpha_mask_from_x_coverage(s, &right_x_cov);

    for tile_y in tile_start_y..tile_end_y {
        let strip_y = tile_y * Tile::HEIGHT;
        let strip_y_f = strip_y as f32;
        let strip_y_end_f = strip_y as f32 + Tile::HEIGHT as f32;

        // A row is an "edge" if the rect's top or bottom boundary falls
        // *inside* it (i.e. partial vertical coverage).
        let is_top_edge = strip_y_f < rect_y0 && rect_y0 < strip_y_end_f;
        let is_bottom_edge = strip_y_f < rect_y1 && rect_y1 < strip_y_end_f;

        if is_top_edge || is_bottom_edge {
            let alpha_start = alpha_buf.len() as u32;

            let y_cov = coverage(strip_y, rect_y0, rect_y1);
            let mut col = left_tile_x;
            // TODO: Can this result in an infinite loop in case x_end == u16::MAX?
            while col + Tile::WIDTH <= x_end {
                // TODO: We could optimize this so this is only computed for the left-most and right-most
                // tile of the edge, all intermediate tiles have full horizontal coverage.
                let x_cov = coverage(col, rect_x0, rect_x1);
                let combined = combined_tile_alpha(s, &x_cov, &y_cov);
                alpha_buf.extend_from_slice(combined.as_slice());
                col += Tile::WIDTH;
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
    let last_strip_y = (tile_end_y - 1) * Tile::HEIGHT;
    strip_buf.push(Strip::new(
        u16::MAX,
        last_strip_y,
        alpha_buf.len() as u32,
        false,
    ));
}

/// Compute fractional pixel coverage for `N` consecutive pixels starting at `start`.
#[inline(always)]
fn coverage<const N: usize>(start: u16, rect_lo: f32, rect_hi: f32) -> [f32; N] {
    let mut cov = [0.0_f32; N];

    #[allow(clippy::needless_range_loop, reason = "better clarity")]
    for i in 0..N {
        let px = (start as usize + i) as f32;
        cov[i] = (rect_hi.min(px + 1.0) - rect_lo.max(px)).clamp(0.0, 1.0);
    }
    cov
}

/// Build an alpha mask for the 4x4 tile from the given horizontal coverages,
/// splatting them across the other dimension.
#[inline(always)]
fn alpha_mask_from_x_coverage<S: Simd>(s: S, cov: &[f32; Tile::WIDTH as usize]) -> u8x16<S> {
    let mut buf = [0_u8; 16];

    #[allow(clippy::needless_range_loop, reason = "better clarity")]
    for col in 0..Tile::WIDTH as usize {
        let alpha = (cov[col] * 255.0 + 0.5) as u8;
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
            buf[col * Tile::HEIGHT as usize + row] = (xc * yc * 255.0 + 0.5) as u8;
        }
    }

    u8x16::from_slice(s, &buf)
}
