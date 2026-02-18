// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Rendering strips.

use crate::flatten::Line;
use crate::kurbo::Rect;
#[cfg(not(feature = "std"))]
use crate::kurbo::common::FloatFuncs as _;
use crate::peniko::Fill;
use crate::tile::{Tile, Tiles};
use crate::util::f32_to_u8;
use alloc::vec::Vec;
use fearless_simd::*;

/// A strip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Strip {
    /// The x coordinate of the strip, in user coordinates.
    pub x: u16,
    /// The y coordinate of the strip, in user coordinates.
    pub y: u16,
    /// Packed alpha index and fill gap flag.
    ///
    /// Bit layout (u32):
    /// - bit 31: `fill_gap` (See `Strip::fill_gap()`).
    /// - bits 0..=30: `alpha_idx` (See `Strip::alpha_idx()`).
    packed_alpha_idx_fill_gap: u32,
}

impl Strip {
    /// The bit mask for `fill_gap` packed into `packed_alpha_idx_fill_gap`.
    const FILL_GAP_MASK: u32 = 1 << 31;

    /// Creates a new strip.
    pub fn new(x: u16, y: u16, alpha_idx: u32, fill_gap: bool) -> Self {
        // Ensure `alpha_idx` does not collide with the fill flag bit.
        assert!(
            alpha_idx & Self::FILL_GAP_MASK == 0,
            "`alpha_idx` too large"
        );
        let fill_gap = u32::from(fill_gap) << 31;
        Self {
            x,
            y,
            packed_alpha_idx_fill_gap: alpha_idx | fill_gap,
        }
    }

    /// Return whether the strip is a sentinel strip.
    pub fn is_sentinel(&self) -> bool {
        self.x == u16::MAX
    }

    /// Return the y coordinate of the strip, in strip units.
    pub fn strip_y(&self) -> u16 {
        self.y / Tile::HEIGHT
    }

    /// Returns the alpha index.
    #[inline(always)]
    pub fn alpha_idx(&self) -> u32 {
        self.packed_alpha_idx_fill_gap & !Self::FILL_GAP_MASK
    }

    /// Sets the alpha index.
    ///
    /// Note that the largest value that can be stored in the alpha index is `u32::MAX << 1`, as the
    /// highest bit is reserved for `fill_gap`.
    #[inline(always)]
    pub fn set_alpha_idx(&mut self, alpha_idx: u32) {
        // Ensure `alpha_idx` does not collide with the fill flag bit.
        assert!(
            alpha_idx & Self::FILL_GAP_MASK == 0,
            "`alpha_idx` too large"
        );
        let fill_gap = self.packed_alpha_idx_fill_gap & Self::FILL_GAP_MASK;
        self.packed_alpha_idx_fill_gap = alpha_idx | fill_gap;
    }

    /// Returns whether the gap that lies between this strip and the previous in the same row should be filled.
    #[inline(always)]
    pub fn fill_gap(&self) -> bool {
        (self.packed_alpha_idx_fill_gap & Self::FILL_GAP_MASK) != 0
    }

    /// Sets whether the gap that lies between this strip and the previous in the same row should be filled.
    #[inline(always)]
    pub fn set_fill_gap(&mut self, fill: bool) {
        let fill = u32::from(fill) << 31;
        self.packed_alpha_idx_fill_gap =
            (self.packed_alpha_idx_fill_gap & !Self::FILL_GAP_MASK) | fill;
    }
}

/// Render the tiles stored in `tiles` into the strip and alpha buffer.
pub fn render(
    level: Level,
    tiles: &Tiles,
    strip_buf: &mut Vec<Strip>,
    alpha_buf: &mut Vec<u8>,
    fill_rule: Fill,
    aliasing_threshold: Option<u8>,
    lines: &[Line],
) {
    dispatch!(level, simd => render_impl(simd, tiles, strip_buf, alpha_buf, fill_rule, aliasing_threshold, lines));
}

#[inline(always)]
fn render_impl<S: Simd>(
    s: S,
    tiles: &Tiles,
    strip_buf: &mut Vec<Strip>,
    alpha_buf: &mut Vec<u8>,
    fill_rule: Fill,
    aliasing_threshold: Option<u8>,
    lines: &[Line],
) {
    if tiles.is_empty() {
        return;
    }

    let should_fill = |winding: i32| match fill_rule {
        Fill::NonZero => winding != 0,
        Fill::EvenOdd => winding % 2 != 0,
    };

    // The accumulated tile winding delta. A line that crosses the top edge of a tile
    // increments the delta if the line is directed upwards, and decrements it if goes
    // downwards. Horizontal lines leave it unchanged.
    let mut winding_delta: i32 = 0;

    // The previous tile visited.
    let mut prev_tile = *tiles.get(0);
    // The accumulated (fractional) winding of the tile-sized location we're currently at.
    // Note multiple tiles can be at the same location.
    // Note that we are also implicitly assuming here that the tile height exactly fits into a
    // SIMD vector (i.e. 128 bits).
    let mut location_winding = [f32x4::splat(s, 0.0); Tile::WIDTH as usize];
    // The accumulated (fractional) windings at this location's right edge. When we move to the
    // next location, this is splatted to that location's starting winding.
    let mut accumulated_winding = f32x4::splat(s, 0.0);

    /// A special tile to keep the logic below simple.
    const SENTINEL: Tile = Tile::new(u16::MAX, u16::MAX, 0, 0);

    // The strip we're building.
    let mut strip = Strip::new(
        prev_tile.x * Tile::WIDTH,
        prev_tile.y * Tile::HEIGHT,
        alpha_buf.len() as u32,
        false,
    );

    for (tile_idx, tile) in tiles.iter().copied().chain([SENTINEL]).enumerate() {
        let line = lines[tile.line_idx() as usize];
        let tile_left_x = f32::from(tile.x) * f32::from(Tile::WIDTH);
        let tile_top_y = f32::from(tile.y) * f32::from(Tile::HEIGHT);
        let p0_x = line.p0.x - tile_left_x;
        let p0_y = line.p0.y - tile_top_y;
        let p1_x = line.p1.x - tile_left_x;
        let p1_y = line.p1.y - tile_top_y;

        // Push out the winding as an alpha mask when we move to the next location (i.e., a tile
        // without the same location).
        if !prev_tile.same_loc(&tile) {
            match fill_rule {
                Fill::NonZero => {
                    let p1 = f32x4::splat(s, 0.5);
                    let p2 = f32x4::splat(s, 255.0);

                    #[expect(clippy::needless_range_loop, reason = "dimension clarity")]
                    for x in 0..Tile::WIDTH as usize {
                        let area = location_winding[x];
                        let coverage = area.abs();
                        let mulled = coverage.mul_add(p2, p1);
                        // Note that we are not storing the location winding here but the actual
                        // alpha value as f32, so we reuse the variable as a temporary storage.
                        // Also note that we need the `min` here because the winding can be > 1
                        // and thus the calculated alpha value need to be clamped to 255.
                        location_winding[x] = mulled.min(p2);
                    }
                }
                Fill::EvenOdd => {
                    let p1 = f32x4::splat(s, 0.5);
                    let p2 = f32x4::splat(s, -2.0);
                    let p3 = f32x4::splat(s, 255.0);

                    #[expect(clippy::needless_range_loop, reason = "dimension clarity")]
                    for x in 0..Tile::WIDTH as usize {
                        let area = location_winding[x];
                        let im1 = area.mul_add(p1, p1).floor();
                        let coverage = p2.mul_add(im1, area).abs();
                        let mulled = p3.mul_add(coverage, p1);
                        // TODO: It is possible that, unlike for `NonZero`, we don't need the `min`
                        // here.
                        location_winding[x] = mulled.min(p3);
                    }
                }
            };

            let p1 = s.combine_f32x4(location_winding[0], location_winding[1]);
            let p2 = s.combine_f32x4(location_winding[2], location_winding[3]);

            let mut u8_vals = f32_to_u8(s.combine_f32x8(p1, p2));

            if let Some(aliasing_threshold) = aliasing_threshold {
                u8_vals = s.select_u8x16(
                    u8_vals.simd_ge(u8x16::splat(s, aliasing_threshold)),
                    u8x16::splat(s, 255),
                    u8x16::splat(s, 0),
                );
            }

            alpha_buf.extend_from_slice(u8_vals.as_slice());

            #[expect(clippy::needless_range_loop, reason = "dimension clarity")]
            for x in 0..Tile::WIDTH as usize {
                location_winding[x] = accumulated_winding;
            }
        }

        // Push out the strip if we're moving to a next strip.
        if !prev_tile.same_loc(&tile) && !prev_tile.prev_loc(&tile) {
            debug_assert_eq!(
                (prev_tile.x as u32 + 1) * Tile::WIDTH as u32 - strip.x as u32,
                ((alpha_buf.len() - strip.alpha_idx() as usize) / usize::from(Tile::HEIGHT)) as u32,
                "The number of columns written to the alpha buffer should equal the number of columns spanned by this strip."
            );
            strip_buf.push(strip);

            let is_sentinel = tile_idx == tiles.len() as usize;
            if !prev_tile.same_row(&tile) {
                // Emit a final strip in the row if there is non-zero winding for the sparse fill,
                // or unconditionally if we've reached the sentinel tile to end the path (the
                // `alpha_idx` field is used for width calculations).
                if winding_delta != 0 || is_sentinel {
                    strip_buf.push(Strip::new(
                        u16::MAX,
                        prev_tile.y * Tile::HEIGHT,
                        alpha_buf.len() as u32,
                        should_fill(winding_delta),
                    ));
                }

                winding_delta = 0;
                accumulated_winding = f32x4::splat(s, 0.0);

                #[expect(clippy::needless_range_loop, reason = "dimension clarity")]
                for x in 0..Tile::WIDTH as usize {
                    location_winding[x] = accumulated_winding;
                }
            }

            if is_sentinel {
                break;
            }

            strip = Strip::new(
                tile.x * Tile::WIDTH,
                tile.y * Tile::HEIGHT,
                alpha_buf.len() as u32,
                should_fill(winding_delta),
            );
            // Note: this fill is mathematically not necessary. It provides a way to reduce
            // accumulation of float rounding errors.
            accumulated_winding = f32x4::splat(s, winding_delta as f32);
        }
        prev_tile = tile;

        // TODO: horizontal geometry has no impact on winding. This branch will be removed when
        // horizontal geometry is culled at the tile-generation stage.
        if p0_y == p1_y {
            continue;
        }

        // Lines moving upwards (in a y-down coordinate system) add to winding; lines moving
        // downwards subtract from winding.
        let sign = (p0_y - p1_y).signum();

        // Calculate winding / pixel area coverage.
        //
        // Conceptually, horizontal rays are shot from left to right. Every time the ray crosses a
        // line that is directed upwards (decreasing `y`), the winding is incremented. Every time
        // the ray crosses a line moving downwards (increasing `y`), the winding is decremented.
        // The fractional area coverage of a pixel is the integral of the winding within it.
        //
        // Practically, to calculate this, each pixel is considered individually, and we determine
        // whether the line moves through this pixel. The line's y-delta within this pixel is
        // accumulated and added to the area coverage of pixels to the right. Within the pixel
        // itself, the area to the right of the line segment forms a trapezoid (or a triangle in
        // the degenerate case). The area of this trapezoid is added to the pixel's area coverage.
        //
        // For example, consider the following pixel square, with a line indicated by asterisks
        // starting inside the pixel and crossing its bottom edge. The area covered is the
        // trapezoid on the bottom-right enclosed by the line and the pixel square. The area is
        // positive if the line moves down, and negative otherwise.
        //
        //  __________________
        //  |                |
        //  |         *------|
        //  |        *       |
        //  |       *        |
        //  |      *         |
        //  |     *          |
        //  |    *           |
        //  |___*____________|
        //     *
        //    *

        let (line_top_y, line_top_x, line_bottom_y, line_bottom_x) = if p0_y < p1_y {
            (p0_y, p0_x, p1_y, p1_x)
        } else {
            (p1_y, p1_x, p0_y, p0_x)
        };

        let (line_left_x, line_left_y, line_right_x) = if p0_x < p1_x {
            (p0_x, p0_y, p1_x)
        } else {
            (p1_x, p1_y, p0_x)
        };

        let y_slope = (line_bottom_y - line_top_y) / (line_bottom_x - line_top_x);
        let x_slope = 1. / y_slope;

        winding_delta += sign as i32 * i32::from(tile.winding());

        // TODO: this should be removed when out-of-viewport tiles are culled at the
        // tile-generation stage. That requires calculating and forwarding winding to strip
        // generation.
        if tile.x == 0 && line_left_x < 0. {
            let (ymin, ymax) = if line.p0.x == line.p1.x {
                (line_top_y, line_bottom_y)
            } else {
                let line_viewport_left_y = (line_top_y - line_top_x * y_slope)
                    .max(line_top_y)
                    .min(line_bottom_y);

                (
                    f32::min(line_left_y, line_viewport_left_y),
                    f32::max(line_left_y, line_viewport_left_y),
                )
            };

            let ymin: f32x4<_> = ymin.simd_into(s);
            let ymax: f32x4<_> = ymax.simd_into(s);

            let px_top_y: f32x4<_> = [0.0, 1.0, 2.0, 3.0].simd_into(s);
            let px_bottom_y = 1.0 + px_top_y;
            let ymin = px_top_y.max(ymin);
            let ymax = px_bottom_y.min(ymax);
            let h = (ymax - ymin).max(0.0);
            accumulated_winding = h.mul_add(sign, accumulated_winding);
            for x_idx in 0..Tile::WIDTH {
                location_winding[x_idx as usize] =
                    h.mul_add(sign, location_winding[x_idx as usize]);
            }

            if line_right_x < 0. {
                // Early exit, as no part of the line is inside the tile.
                continue;
            }
        }

        let line_top_y = f32x4::splat(s, line_top_y);
        let line_bottom_y = f32x4::splat(s, line_bottom_y);

        // See the explanation of this term on the `line_px_left_yx` and `line_px_right_yx`
        // variables below.
        let line_px_base_yx = line_top_y.mul_add(-x_slope, line_top_x);

        let px_top_y = f32x4::simd_from(s, [0., 1., 2., 3.]);
        let px_bottom_y = 1. + px_top_y;

        let ymin = line_top_y.max(px_top_y);
        let ymax = line_bottom_y.min(px_bottom_y);

        let mut acc = f32x4::splat(s, 0.0);

        for x_idx in 0..Tile::WIDTH {
            let x_idx_s = f32x4::splat(s, x_idx as f32);
            let px_left_x = x_idx_s;
            let px_right_x = 1.0 + x_idx_s;

            // The y-coordinate of the intersections between the line and the pixel's left and
            // right edges respectively.
            //
            // There is some subtlety going on here: `y_slope` will usually be finite, but will
            // be `inf` for purely vertical lines (`p0_x == p1_x`).
            //
            // In the case of `inf`, the resulting slope calculation will be `-inf` or `inf`
            // depending on whether the pixel edge is left or right of the line, respectively
            // (from the viewport's coordinate system perspective). The `min` and `max`
            // y-clamping logic generalizes nicely, as a pixel edge to the left of the line is
            // clamped to `ymin`, and a pixel edge to the right is clamped to `ymax`.
            //
            // In the special case where a vertical line and pixel edge are at the exact same
            // x-position (collinear), the line belongs to the pixel on whose _left_ edge it is
            // situated. The resulting slope calculation for the edge the line is situated on
            // will be NaN, as `0 * inf` results in NaN. This is true for both the left and
            // right edge.
            //
            // We know `ymin` and `ymax` are finite. We require the `max` operation to pick `ymin`
            // if its first operand is NaN. Under a strict reading of `fearless_simd`'s `max` and
            // `max_precise` semantics, that requires using `max_precise`, which chooses the
            // non-NaN operand (regardless of whether it's the first or second). For `min`, we then
            // know both operands are finite, so we can use the relaxed version.
            let line_px_left_y = (px_left_x - line_top_x)
                .mul_add(y_slope, line_top_y)
                .max_precise(ymin)
                .min(ymax);
            let line_px_right_y = (px_right_x - line_top_x)
                .mul_add(y_slope, line_top_y)
                .max_precise(ymin)
                .min(ymax);

            // For each pixel we calculate the x-coordinates of the left- and rightmost points on
            // the line segment within that pixel. We do this based on the y-offsets of those two
            // points from the top of the line. This can be calculated as, e.g.,
            // `(line_px_left_y - line_top_y) * x_slope + line_top_x`.
            //
            // Rather than calculating that y-offset twice for each pixel within the loop through
            // subtracting from the points' y-coordinates, we get rid of that subtraction by baking
            // it into the algebraic "base" x-coordinate `line_px_base_yx` calculated above the
            // loop. When adding that term to `y * x_slope` it gives the x-coordinate of the point
            // along the line.
            //
            // Note `x_slope` is always finite, as horizontal geometry is elided.
            let line_px_left_yx = line_px_left_y.mul_add(x_slope, line_px_base_yx);
            let line_px_right_yx = line_px_right_y.mul_add(x_slope, line_px_base_yx);
            let h = (line_px_right_y - line_px_left_y).abs();

            // The trapezoidal area enclosed between the line and the right edge of the pixel
            // square. More straightforwardly written as follows, but the `madd` is faster.
            // 0.5 * h * (2. * px_right_x - line_px_right_yx - line_px_left_yx).
            let area = h * (line_px_right_yx + line_px_left_yx).mul_add(-0.5, px_right_x);
            location_winding[x_idx as usize] += area.mul_add(sign, acc);
            acc = h.mul_add(sign, acc);
        }

        accumulated_winding += acc;
    }
}

/// Render a pixel-aligned rectangle directly into strips.
///
/// This bypasses the full path processing pipeline (flatten → tiles → strips)
/// by directly creating strip coverage data for the rectangle.
///
/// The rect bounds should already be clamped to the viewport.
pub fn render_rect_fast(
    level: Level,
    rect: Rect,
    strip_buf: &mut Vec<Strip>,
    alpha_buf: &mut Vec<u8>,
) {
    dispatch!(level, simd => render_rect_fast_impl(simd, rect, strip_buf, alpha_buf));
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
fn render_rect_fast_impl<S: Simd>(
    s: S,
    rect: Rect,
    strip_buf: &mut Vec<Strip>,
    alpha_buf: &mut Vec<u8>,
) {
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

    let (max_strips, max_alphas) =
        calc_rect_reserve_capacity(x_end - left_tile_x, tile_end_y - tile_start_y);
    strip_buf.reserve(max_strips);
    alpha_buf.reserve(max_alphas);

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

/// Calculate the maximum buffer capacity needed for rendering a pixel-aligned rect.
///
/// Returns `(max_strips, max_alphas)` based on the tile dimensions covered.
#[inline]
fn calc_rect_reserve_capacity(x_span: u16, tile_rows: u16) -> (usize, usize) {
    let tile_cols = (x_span / Tile::WIDTH) as usize;
    let tile_rows = tile_rows as usize;
    // Max strips: 2 per row (left + right) + 1 sentinel
    let max_strips = tile_rows * 2 + 1;
    // Max alphas: worst case is edge rows covering all columns
    // Each column needs Tile::HEIGHT bytes
    let max_alphas = tile_cols * tile_rows * Tile::WIDTH as usize * Tile::HEIGHT as usize;
    (max_strips, max_alphas)
}
