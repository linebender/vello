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
// TODO: Consider extending this to handle arbitrary axis-aligned rectangles (with fractional
// coordinates) by computing partial coverage alpha values for edge pixels instead of
// binary 0/255.
fn render_impl<S: Simd>(s: S, rect: Rect, strip_buf: &mut Vec<Strip>, alpha_buf: &mut Vec<u8>) {
    // TODO: Negative rect coordinates are not handled correctly — casting negative
    // f64 to u16 saturates to 0. The caller currently clamps to the viewport, but if
    // that changes, this will need signed-integer math or explicit clamping.
    let rect_x0 = rect.x0.floor() as u16;
    let rect_y0 = rect.y0.floor() as u16;
    let rect_x1 = rect.x1.ceil() as u16;
    let rect_y1 = rect.y1.ceil() as u16;

    let left_tile_x = (rect_x0 / Tile::WIDTH) * Tile::WIDTH;
    let right_tile_x = (rect_x1 / Tile::WIDTH) * Tile::WIDTH;

    let y0 = (rect_y0 / Tile::HEIGHT) * Tile::HEIGHT;
    let y1 = (rect_y1.saturating_add(Tile::HEIGHT - 1) / Tile::HEIGHT) * Tile::HEIGHT;
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

    let left_x_mask = x_alpha_tile(s, left_tile_x, rect_x0, rect_x1);
    let right_x_mask = x_alpha_tile(s, right_tile_x, rect_x0, rect_x1);

    for tile_y in tile_start_y..tile_end_y {
        let strip_y = tile_y * Tile::HEIGHT;

        // A row is an "edge" if the rect's top or bottom boundary falls
        // *inside* it (i.e. partial vertical coverage).
        let is_top_edge = strip_y < rect_y0 && rect_y0 < strip_y + Tile::HEIGHT;
        let is_bottom_edge = strip_y < rect_y1 && rect_y1 < strip_y + Tile::HEIGHT;

        if is_top_edge || is_bottom_edge {
            let alpha_start = alpha_buf.len() as u32;
            let y_mask = y_alpha_tile(s, strip_y, rect_y0, rect_y1);

            // Walk every tile column, AND the per-column x-mask with the
            // per-row y-mask to get the final per-pixel alpha.
            let mut col = left_tile_x;
            while col + Tile::WIDTH <= x_end {
                let combined = x_alpha_tile(s, col, rect_x0, rect_x1) & y_mask;
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

/// Build a column-major x-alpha mask for one tile-width of columns.
///
/// Each column gets `Tile::HEIGHT` lanes, all 0x00 or all 0xFF depending on
/// whether the column falls inside `[rect_x0, rect_x1)`.
#[inline(always)]
fn x_alpha_tile<S: Simd>(s: S, tile_x: u16, rect_x0: u16, rect_x1: u16) -> u8x16<S> {
    let mut buf = [0_u8; 16];
    for col in 0..Tile::WIDTH {
        let px = tile_x + col;
        let alpha = if px >= rect_x0 && px < rect_x1 {
            255
        } else {
            0
        };
        let base = (col * Tile::HEIGHT) as usize;
        buf[base..base + Tile::HEIGHT as usize].fill(alpha);
    }
    u8x16::from_slice(s, &buf)
}

/// Build a column-major y-alpha mask for one tile row.
///
/// Each of the `Tile::HEIGHT` rows is 0x00 or 0xFF depending on whether it
/// falls inside `[rect_y0, rect_y1)`. The pattern is identical across all
/// `Tile::WIDTH` columns.
#[inline(always)]
fn y_alpha_tile<S: Simd>(s: S, strip_y: u16, rect_y0: u16, rect_y1: u16) -> u8x16<S> {
    let mut y_mask = [0_u8; 4];
    for row in 0..Tile::HEIGHT {
        let py = strip_y + row;
        y_mask[row as usize] = if py >= rect_y0 && py < rect_y1 {
            255
        } else {
            0
        };
    }
    let mut buf = [0_u8; 16];
    for col in 0..Tile::WIDTH as usize {
        let base = col * Tile::HEIGHT as usize;
        buf[base..base + Tile::HEIGHT as usize].copy_from_slice(&y_mask);
    }
    u8x16::from_slice(s, &buf)
}
