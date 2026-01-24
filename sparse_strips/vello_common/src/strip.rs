// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Rendering strips.

use core::arch::aarch64::{uint32x2x4_t, uint32x4x2_t};

use crate::flatten::Line;
use crate::peniko::Fill;
use crate::tile::{Tile, Tiles};
use crate::util::f32_to_u8;
use alloc::fmt::Debug;
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
pub fn render<T: MsaaMask>(
    level: Level,
    tiles: &Tiles,
    strip_buf: &mut Vec<Strip>,
    alpha_buf: &mut Vec<u8>,
    fill_rule: Fill,
    aliasing_threshold: Option<u8>,
    lines: &[Line],
    mask_lut: &[T],
) {
    if mask_lut.is_empty() {
        dispatch!(level, simd => render_analytic_impl(simd,
                                                      tiles,
                                                      strip_buf,
                                                      alpha_buf,
                                                      fill_rule,
                                                      aliasing_threshold,
                                                      lines));
    } else {
        dispatch!(level, simd => T::render_msaa(simd,
                                                tiles,
                                                strip_buf,
                                                alpha_buf,
                                                fill_rule,
                                                aliasing_threshold,
                                                lines,
                                                mask_lut));
    }
}

fn render_analytic_impl<S: Simd>(
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
                        let mulled = coverage.madd(p2, p1);
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
                        let im1 = area.madd(p1, p1).floor();
                        let coverage = p2.madd(im1, area).abs();
                        let mulled = p3.madd(coverage, p1);
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
            accumulated_winding = h.madd(sign, accumulated_winding);
            for x_idx in 0..Tile::WIDTH {
                location_winding[x_idx as usize] = h.madd(sign, location_winding[x_idx as usize]);
            }

            if line_right_x < 0. {
                // Early exit, as no part of the line is inside the tile.
                continue;
            }
        }

        let line_top_y = f32x4::splat(s, line_top_y);
        let line_bottom_y = f32x4::splat(s, line_bottom_y);

        let y_idx = f32x4::from_slice(s, &[0.0, 1.0, 2.0, 3.0]);
        let px_top_y = y_idx;
        let px_bottom_y = 1. + y_idx;

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
            // right edge. In both cases, the call to `f32::max` will set this to `ymin`.
            let line_px_left_y = (px_left_x - line_top_x)
                .madd(y_slope, line_top_y)
                .max_precise(ymin)
                .min_precise(ymax);
            let line_px_right_y = (px_right_x - line_top_x)
                .madd(y_slope, line_top_y)
                .max_precise(ymin)
                .min_precise(ymax);

            // `x_slope` is always finite, as horizontal geometry is elided.
            let line_px_left_yx =
                (line_px_left_y - line_top_y).madd(x_slope, f32x4::splat(s, line_top_x));
            let line_px_right_yx =
                (line_px_right_y - line_top_y).madd(x_slope, f32x4::splat(s, line_top_x));
            let h = (line_px_right_y - line_px_left_y).abs();

            // The trapezoidal area enclosed between the line and the right edge of the pixel
            // square.
            let area = 0.5 * h * (2. * px_right_x - line_px_right_yx - line_px_left_yx);
            location_winding[x_idx as usize] += area.madd(sign, acc);
            acc = h.madd(sign, acc);
        }

        accumulated_winding += acc;
    }
}

const MASK_WIDTH: u32 = 64;
const MASK_HEIGHT: u32 = 64;
const PACKING_SCALE: f32 = 0.5;

fn generate_mask_lut_generic<const N: usize, T: MsaaMask>(pattern: [u8; N]) -> Vec<T> {
    let scale = 1.0 / (N as f32);

    let mut sub_x = [0.0; N];
    let mut sub_y = [0.0; N];

    for k in 0..N {
        sub_x[k] = (pattern[k] as f32 + 0.5) * scale;
        sub_y[k] = (k as f32 + 0.5) * scale;
    }

    let mut lut = Vec::with_capacity((MASK_WIDTH * MASK_HEIGHT) as usize);

    for j in 0..MASK_WIDTH {
        for i in 0..MASK_HEIGHT {
            let xf = (i as f32 + 0.5) / MASK_WIDTH as f32;
            let yf = (j as f32 + 0.5) / MASK_HEIGHT as f32;

            let n_rev = (2.0 * (xf - 0.5), 2.0 * (yf - 0.5));
            let mut lg_rev = n_rev.0.hypot(n_rev.1);
            if lg_rev < 1e-9 {
                lg_rev = 1e-9;
            }

            let n_lookup = (n_rev.0 / lg_rev, n_rev.1 / lg_rev);
            let c_dist_unsigned = (1.0 - lg_rev).max(0.0) * (1.0 / PACKING_SCALE);

            let mut n_canonical = n_lookup;
            let mut c_signed_dist = c_dist_unsigned;

            if n_lookup.0 < 0.0 {
                n_canonical.0 = -n_lookup.0;
                n_canonical.1 = -n_lookup.1;
                c_signed_dist = -c_dist_unsigned;
            }

            let c_plane = c_signed_dist + 0.5 * (n_canonical.0 + n_canonical.1);

            let mut mask = T::default();
            for k in 0..N {
                let p = (sub_x[k], sub_y[k]);
                if n_canonical.0 * p.0 + n_canonical.1 * p.1 - c_plane > 0.0 {
                    mask.set_bit(k);
                }
            }
            lut.push(mask);
        }
    }
    lut
}

///askjbd
pub fn generate_mask_lut_msaa8() -> Vec<u8> {
    const PATTERN: [u8; 8] = [0, 5, 3, 7, 1, 4, 6, 2];
    generate_mask_lut_generic::<8, u8>(PATTERN)
}

///asidbasiudu
pub fn generate_mask_lut_msaa16() -> Vec<u16> {
    const PATTERN_16: [u8; 16] = [1, 8, 4, 11, 15, 7, 3, 12, 0, 9, 5, 13, 2, 10, 6, 14];
    generate_mask_lut_generic::<16, u16>(PATTERN_16)
}

fn algorithm_lut_mask_halfplane(line_p0: [f32; 2], line_p1: [f32; 2], lut: &[u8]) -> u8 {
    let p0 = line_p0;
    let p1 = line_p1;

    let dir = (p1[0] - p0[0], p1[1] - p0[1]);
    let n_unnormalized = (dir.1, -dir.0);

    let mut len = n_unnormalized.0.hypot(n_unnormalized.1);
    if len < 1e-9 {
        len = 1e-9;
    }

    let n = (n_unnormalized.0 / len, n_unnormalized.1 / len);
    let mut c = n.0 * p0[0] + n.1 * p0[1];
    c -= 0.5 * (n.0 + n.1);

    let c_lookup = c;
    let sign = if c < 0.0 { -1.0 } else { 1.0 };

    let c2 = (1.0 - c_lookup * PACKING_SCALE * sign).max(0.0);

    let n_rev = (c2 * n.0 * sign, c2 * n.1 * sign);

    let mut uv = (n_rev.0 * 0.5 + 0.5, n_rev.1 * 0.5 + 0.5);

    if sign < 0.0 && uv.0 == 0.5 {
        uv.0 = 0.5f32.next_down();
    }

    let u_f = (uv.0 * MASK_WIDTH as f32).floor();
    let v_f = (uv.1 * MASK_HEIGHT as f32).floor();

    let mask_width_m1 = (MASK_WIDTH - 1) as i32;
    let mask_height_m1 = (MASK_HEIGHT - 1) as i32;

    let u = (u_f as i32).max(0).min(mask_width_m1) as u32;
    let v = (v_f as i32).max(0).min(mask_height_m1) as u32;

    let index = (v * MASK_WIDTH + u) as usize;
    lut[index]
}

/// Clips a line segment to a tile.
///
/// Returns the start and end points of the clipped line segment relative to the tile origin.
pub fn clip_to_tile(
    line: &Line,
    bounds: &[f32; 4],
    derivatives: &[f32; 4],
    intersection_data: u32,
    cannonical_x_dir: bool,
    cannonical_y_dir: bool,
) -> [[f32; 2]; 2] {
    const INTERSECTS_TOP_MASK: u32 = 1;
    const INTERSECTS_BOTTOM_MASK: u32 = 2;
    const INTERSECTS_LEFT_MASK: u32 = 4;
    const INTERSECTS_RIGHT_MASK: u32 = 8;
    const PERFECT_MASK: u32 = 16;

    let mut p_entry = [line.p0.x, line.p0.y];
    let mut p_exit = [line.p1.x, line.p1.y];

    let tile_min_x = bounds[0];
    let tile_min_y = bounds[1];
    let tile_max_x = bounds[2];
    let tile_max_y = bounds[3];

    let dx = derivatives[0];
    let dy = derivatives[1];

    let (mask_v_in, bound_v_in, mask_v_out, bound_v_out) = if cannonical_x_dir {
        (
            INTERSECTS_LEFT_MASK,
            tile_min_x,
            INTERSECTS_RIGHT_MASK,
            tile_max_x,
        )
    } else {
        (
            INTERSECTS_RIGHT_MASK,
            tile_max_x,
            INTERSECTS_LEFT_MASK,
            tile_min_x,
        )
    };

    let (mask_h_in, bound_h_in, mask_h_out, bound_h_out) = if cannonical_y_dir {
        (
            INTERSECTS_TOP_MASK,
            tile_min_y,
            INTERSECTS_BOTTOM_MASK,
            tile_max_y,
        )
    } else {
        (
            INTERSECTS_BOTTOM_MASK,
            tile_max_y,
            INTERSECTS_TOP_MASK,
            tile_min_y,
        )
    };

    let idx = derivatives[2];
    let idy = derivatives[3];

    let entry_hits = intersection_data & (mask_v_in | mask_h_in);
    if entry_hits != 0 {
        let use_h = (intersection_data & mask_h_in) != 0;

        let bound = if use_h { bound_h_in } else { bound_v_in };
        let start = if use_h { line.p0.y } else { line.p0.x };
        let inv_d = if use_h { idy } else { idx };

        let t = (bound - start) * inv_d;

        p_entry[0] = line.p0.x + t * dx;
        p_entry[1] = line.p0.y + t * dy;
        p_entry[if use_h { 1 } else { 0 }] = bound;
    }

    let exit_hits = intersection_data & (mask_v_out | mask_h_out);
    if exit_hits != 0 {
        let use_h = (intersection_data & mask_h_out) != 0;

        let bound = if use_h { bound_h_out } else { bound_v_out };
        let start = if use_h { line.p0.y } else { line.p0.x };
        let inv_d = if use_h { idy } else { idx };

        let t = (bound - start) * inv_d;

        p_exit[0] = line.p0.x + t * dx;
        p_exit[1] = line.p0.y + t * dy;
        p_exit[if use_h { 1 } else { 0 }] = bound;
    }

    let mut result = if p_exit[1] >= p_entry[1] {
        [p_entry, p_exit]
    } else {
        [p_exit, p_entry]
    };

    result[0][0] -= tile_min_x;
    result[0][1] -= tile_min_y;
    result[1][0] -= tile_min_x;
    result[1][1] -= tile_min_y;

    // Clamping has a dual purpose here:
    // 1) Points which are slightly outside the tile due to floating point error are coerced inside.
    // 2) More subtly, perfectly horizontal or vertical lines have their reciprocal derivatives
    //    set to 0. This causes the intersection calculation to return the original coordinate.
    //    While the coordinate fixed to the tile edge is explicitly set (and guaranteed valid),
    //    clamping forces the coordinate along that edge to be in bounds and watertight.
    let width = Tile::WIDTH as f32;
    let height = Tile::HEIGHT as f32;
    result[0][0] = result[0][0].clamp(0.0, width);
    result[0][1] = result[0][1].clamp(0.0, height);
    result[1][0] = result[1][0].clamp(0.0, width);
    result[1][1] = result[1][1].clamp(0.0, height);

    result
}

///aaaaa
pub trait MsaaMask: Copy + Default + Debug + 'static {
    /// Sets a bit in the mask. Used during LUT generation.
    fn set_bit(&mut self, bit: usize);

    /// Renders using this mask type.
    fn render_msaa<S: Simd>(
        simd: S,
        tiles: &Tiles,
        strip_buf: &mut Vec<Strip>,
        alpha_buf: &mut Vec<u8>,
        fill_rule: Fill,
        aliasing_threshold: Option<u8>,
        lines: &[Line],
        mask_lut: &[Self],
    );
}

impl MsaaMask for u8 {
    fn set_bit(&mut self, bit: usize) {
        *self |= 1 << bit;
    }

    fn render_msaa<S: Simd>(
        s: S,
        tiles: &Tiles,
        strip_buf: &mut Vec<Strip>,
        alpha_buf: &mut Vec<u8>,
        fill_rule: Fill,
        aliasing_threshold: Option<u8>,
        lines: &[Line],
        mask_lut: &[Self],
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

        // Winding Counts
        let mut mask = [u32x8::splat(s, 0x80808080u32); Tile::HEIGHT as usize];

        /// A special tile to keep the logic below simple.
        const SENTINEL: Tile = Tile::new(u16::MAX, u16::MAX, 0, 0);

        // The strip we're building.
        let mut strip = Strip::new(
            prev_tile.x * Tile::WIDTH,
            prev_tile.y * Tile::HEIGHT,
            alpha_buf.len() as u32,
            false,
        );

        let tile_min_x_u32 = (prev_tile.x as u32) * (Tile::WIDTH as u32);
        let tile_min_y_u32 = (prev_tile.y as u32) * (Tile::HEIGHT as u32);

        let mut tile_min_x_px = tile_min_x_u32 as f32;
        let mut tile_min_y_px = tile_min_y_u32 as f32;
        let mut tile_max_x_px = (tile_min_x_u32 + (Tile::WIDTH as u32)) as f32;
        let mut tile_max_y_px = (tile_min_y_u32 + (Tile::HEIGHT as u32)) as f32;

        for (tile_idx, tile) in tiles.iter().copied().chain([SENTINEL]).enumerate() {
            let tile_start = !prev_tile.same_loc(&tile);
            let row_start = !prev_tile.same_row(&tile);
            let seg_start = tile_start && !prev_tile.prev_loc(&tile);

            if tile_start {
                if tile.x > 0 {
                    let bias = u32x8::splat(s, 0x80808080u32);
                    let ones = u32x8::splat(s, 0x01010101u32);
                    let scale_mul = u32x8::splat(s, 255);
                    let scale_add = u32x8::splat(s, 4);


                    let mut computed_rows = [u32x8::splat(s, 0); Tile::HEIGHT as usize];
                    for y in 0..Tile::HEIGHT as usize {
                        let v = mask[y];
                        let diff = v.xor(bias);
                        let subbed = diff.sub(ones);
                        let zero_markers = subbed.and(diff.not()).and(bias);
                        let active_markers = zero_markers.xor(bias);
                        let counts = active_markers.shr(7).mul(ones).shr(24);
                        let evens = counts.unzip_low(counts);
                        let odds = counts.unzip_high(counts);
                        let pixel_counts = evens.add(odds);
                        computed_rows[y] = pixel_counts.mul(scale_mul).add(scale_add).shr(3);
                    }

                    // Transpose, wasteful?
                    for x in 0..4 {
                        for y in 0..Tile::HEIGHT as usize {
                            let val = computed_rows[y].as_slice()[x];
                            alpha_buf.push(val as u8);
                        }
                    }
                } else {
                    alpha_buf
                        .extend(core::iter::repeat(0).take((Tile::WIDTH * Tile::HEIGHT) as usize));
                }

                if !row_start {
                    let w = u32x8::splat(
                        s,
                        0x80808080u32
                            .wrapping_add((winding_delta as i8 as u32).wrapping_mul(0x01010101u32)),
                    );
                    mask.fill(w);
                }
            }

            // Push out the strip if we're moving to a next strip.
            if seg_start {
                debug_assert_eq!(
                    (prev_tile.x as u32 + 1) * Tile::WIDTH as u32 - strip.x as u32,
                    ((alpha_buf.len() - strip.alpha_idx() as usize) / usize::from(Tile::HEIGHT))
                        as u32,
                    "The number of columns written to the alpha buffer should equal the number of columns spanned by this strip."
                );
                strip_buf.push(strip);

                let is_sentinel = tile_idx == tiles.len() as usize;
                if row_start {
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
                    #[expect(clippy::needless_range_loop, reason = "dimension clarity")]
                    for y in 0..Tile::HEIGHT as usize {
                        mask[y] = u32x8::splat(s, 0x80808080u32);
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
            }
            prev_tile = tile;

            let line = lines[tile.line_idx() as usize];

            let p0_x = line.p0.x;
            let p0_y = line.p0.y;
            let p1_x = line.p1.x;
            let p1_y = line.p1.y;

            let canonical_x_dir = p1_x >= p0_x;
            let canonical_y_dir = p1_y >= p0_y;

            let sign = if canonical_y_dir { 1i32 } else { -1i32 };
            let signed_winding = sign * tile.winding() as i32;
            winding_delta += signed_winding;

            // If the rightmost point of a line is left of the viewport, no further processing is
            // required. The coarse mask is the only dependency which need be passed on.
            let right = if canonical_x_dir { p1_x } else { p0_x };
            let right_in_viewport = right >= 0.0;
            if !right_in_viewport {
                continue;
            }

            let dx = p1_x - p0_x;
            let dy = p1_y - p0_y;
            let is_vertical = dx.abs() <= f32::EPSILON;
            let is_horizontal = dy.abs() <= f32::EPSILON;
            let idx = if is_vertical { 0.0 } else { 1.0 / dx };
            let idy = if is_horizontal { 0.0 } else { 1.0 / dy };
            let dxdy = dx * idy;

            if tile_start {
                let min_x_u32 = (tile.x as u32) * (Tile::WIDTH as u32);
                let min_y_u32 = (tile.y as u32) * (Tile::HEIGHT as u32);

                tile_min_x_px = min_x_u32 as f32;
                tile_min_y_px = min_y_u32 as f32;
                tile_max_x_px = (min_x_u32 + (Tile::WIDTH as u32)) as f32;
                tile_max_y_px = (min_y_u32 + (Tile::HEIGHT as u32)) as f32;
            }

            let derivatives = [dx, dy, idx, idy];
            let [clipped_top, clipped_bot] = clip_to_tile(
                &line,
                &[tile_min_x_px, tile_min_y_px, tile_max_x_px, tile_max_y_px],
                &derivatives,
                tile.intersection_mask(),
                canonical_x_dir,
                canonical_y_dir,
            );

            let left = tile.intersects_left();
            if left {
                let y_edge = if clipped_top[0] <= clipped_bot[0] {
                    clipped_top[1]
                } else {
                    clipped_bot[1]
                };
                let v = if canonical_x_dir {
                    u32x8::splat(s, 0xfefefeffu32)
                } else {
                    u32x8::splat(s, 0x1010101u32)
                };
                let y_cross = y_edge.ceil() as usize;
                for y in y_cross..Tile::HEIGHT as usize {
                    mask[y] += v;
                }
            }

            // Discard perfectly axis aligned horizontal lines as the vertical mask produces the correct
            // value. This also ensures that the ceil call for end_y will always produce a value
            // distinct from start_y.
            if is_horizontal && clipped_top[1] == clipped_top[1].floor() {
                continue;
            }

            let start_y = clipped_top[1].floor() as usize;
            let end_y = clipped_bot[1].ceil() as usize;
            let mut top_row = [[f32::NAN; 2]; (Tile::HEIGHT + 1) as usize];
            {
                top_row[start_y] = clipped_top;
                for y_idx in (start_y + 1)..end_y {
                    let grid_y = y_idx as f32;
                    let grid_x = clipped_top[0] + (grid_y - clipped_top[1]) * dxdy;
                    top_row[y_idx] = [grid_x, grid_y];
                }
                top_row[end_y] = clipped_bot;
            }

            let x_dir = clipped_top[0] <= clipped_bot[0];
            for y in start_y..end_y {
                let p_top = top_row[y];
                let p_bottom = top_row[y + 1];

                if p_top[0].is_nan() || p_bottom[0].is_nan() {
                    continue;
                }

                let x_min = p_top[0].min(p_bottom[0]);
                let x_max = p_top[0].max(p_bottom[0]);
                let x_start = (x_min.floor() as i32).clamp(0, Tile::WIDTH as i32 - 1) as usize;
                let x_end = (x_max.floor() as i32).clamp(0, Tile::WIDTH as i32 - 1) as usize;

                let py = y as f32;
                let dy_top = p_top[1] - py;
                let dy_bot = p_bottom[1] - py;
                let pixel_top_touch = p_top[1] != p_top[1].floor();

                let mask_shift_top = (8.0 * dy_top).round() as u32;
                let top_clip_mask = 0xffu32 << mask_shift_top;

                let mask_shift_bot = (8.0 * dy_bot).round() as u32;
                let bot_clip_mask = !(0xffu32 << mask_shift_bot);

                let mut row_mask = [0x80808080u32; Tile::WIDTH as usize * 2];
                for x in x_start..=x_end {
                    let px = x as f32;

                    let mut msaa_mask = algorithm_lut_mask_halfplane(
                        [(p_top[0] - px), dy_top],
                        [(p_bottom[0] - px), dy_bot],
                        &mask_lut,
                    ) as u32;

                    let is_start = x == x_start;
                    let is_end = x == x_end;
                    let canonical_start = (x_dir && is_start) || (!x_dir && is_end);
                    let canonical_end = (x_dir && is_end) || (!x_dir && is_start);
                    let line_top = canonical_start && y == start_y;

                    let bumped = (line_top && p_top[0] == 0.0 && pixel_top_touch)
                        || (!line_top && pixel_top_touch && x_dir);

                    if line_top && !bumped {
                        msaa_mask &= top_clip_mask;
                    }

                    if canonical_end && y == end_y - 1 && p_bottom[0] != 0.0 {
                        msaa_mask &= bot_clip_mask;
                    }

                    let mask_a = msaa_mask ^ (msaa_mask << 7u32);
                    let mask_b = mask_a ^ (mask_a << 14u32);
                    let mut mask_lo = mask_b & 0x1010101u32;
                    let mut mask_hi = mask_b >> 4 & 0x1010101u32;

                    if bumped {
                        let ones = 0x1010101u32;
                        mask_lo = mask_lo.wrapping_sub(ones);
                        mask_hi = mask_hi.wrapping_sub(ones);
                    }

                    if canonical_y_dir {
                        row_mask[x << 1] += mask_lo;
                        row_mask[(x << 1) + 1] += mask_hi;
                    } else {
                        row_mask[x << 1] -= mask_lo;
                        row_mask[(x << 1) + 1] -= mask_hi;
                    }
                }

                let crossed_top = y > start_y || p_top[1] == p_top[1].floor();
                if crossed_top {
                    let f = if canonical_y_dir {
                        0x1010101u32
                    } else {
                        0xfefefeffu32
                    };
                    for x in (x_end + 1)..Tile::WIDTH as usize {
                        row_mask[x << 1] = row_mask[x << 1].wrapping_add(f);
                        row_mask[(x << 1) + 1] = row_mask[(x << 1) + 1].wrapping_add(f);
                    }
                }

                mask[y] += u32x8::from_slice(s, &row_mask) - u32x8::splat(s, 0x80808080u32);
            }
        }
    }
}

impl MsaaMask for u16 {
    fn set_bit(&mut self, bit: usize) {
        *self |= 1 << bit;
    }

    fn render_msaa<S: Simd>(
        _s: S,
        _tiles: &Tiles,
        _strip_buf: &mut Vec<Strip>,
        _alpha_buf: &mut Vec<u8>,
        _fill_rule: Fill,
        _aliasing_threshold: Option<u8>,
        _lines: &[Line],
        _mask_lut: &[Self],
    ) {
        // MSAA16 stub
    }
}
