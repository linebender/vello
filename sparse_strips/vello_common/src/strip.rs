// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Rendering strips.

use crate::flatten::Line;
use crate::peniko::Fill;
use crate::tile::{Tile, Tiles};
use crate::util::f32_to_u8;
use alloc::vec::Vec;
use fearless_simd::*;

/// A strip.
#[derive(Debug, Clone, Copy)]
pub struct Strip {
    /// The x coordinate of the strip, in user coordinates.
    pub x: u16,
    /// The y coordinate of the strip, in user coordinates.
    pub y: u16,
    /// The index into the alpha buffer.
    pub alpha_idx: u32,
    /// The winding number at the start of the strip.
    pub winding: i32,
}

impl Strip {
    /// Return the y coordinate of the strip, in strip units.
    pub fn strip_y(&self) -> u16 {
        self.y / Tile::HEIGHT
    }
}

/// Render the tiles stored in `tiles` into the strip and alpha buffer.
/// The strip buffer will be cleared in the beginning.
pub fn render(
    level: Level,
    tiles: &Tiles,
    strip_buf: &mut Vec<Strip>,
    alpha_buf: &mut Vec<u8>,
    fill_rule: Fill,
    aliasing_threshold: Option<u8>,
    lines: &[Line],
) {
    render_dispatch(
        level,
        tiles,
        strip_buf,
        alpha_buf,
        fill_rule,
        aliasing_threshold,
        lines,
    );
}

simd_dispatch!(fn render_dispatch(
    level,
    tiles: &Tiles,
    strip_buf: &mut Vec<Strip>,
    alpha_buf: &mut Vec<u8>,
    fill_rule: Fill,
    aliasing_threshold: Option<u8>,
    lines: &[Line],
) = render_impl);

fn render_impl<S: Simd>(
    s: S,
    tiles: &Tiles,
    strip_buf: &mut Vec<Strip>,
    alpha_buf: &mut Vec<u8>,
    fill_rule: Fill,
    aliasing_threshold: Option<u8>,
    lines: &[Line],
) {
    strip_buf.clear();

    if tiles.is_empty() {
        return;
    }

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
    const SENTINEL: Tile = Tile::new(u16::MAX, u16::MAX, 0, false);

    // The strip we're building.
    let mut strip = Strip {
        x: prev_tile.x * Tile::WIDTH,
        y: prev_tile.y * Tile::HEIGHT,
        alpha_idx: alpha_buf.len() as u32,
        winding: 0,
    };

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
                        let mulled = p1.madd(coverage, p2);
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
                        let im1 = p1.madd(area, p1).floor();
                        let coverage = area.madd(p2, im1).abs();
                        let mulled = p1.madd(p3, coverage);
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

            alpha_buf.extend_from_slice(&u8_vals.val);

            #[expect(clippy::needless_range_loop, reason = "dimension clarity")]
            for x in 0..Tile::WIDTH as usize {
                location_winding[x] = accumulated_winding;
            }
        }

        // Push out the strip if we're moving to a next strip.
        if !prev_tile.same_loc(&tile) && !prev_tile.prev_loc(&tile) {
            debug_assert_eq!(
                (prev_tile.x + 1) * Tile::WIDTH - strip.x,
                ((alpha_buf.len() - strip.alpha_idx as usize) / usize::from(Tile::HEIGHT)) as u16,
                "The number of columns written to the alpha buffer should equal the number of columns spanned by this strip."
            );
            strip_buf.push(strip);

            let is_sentinel = tile_idx == tiles.len() as usize;
            if !prev_tile.same_row(&tile) {
                // Emit a final strip in the row if there is non-zero winding for the sparse fill,
                // or unconditionally if we've reached the sentinel tile to end the path (the
                // `alpha_idx` field is used for width calculations).
                if winding_delta != 0 || is_sentinel {
                    strip_buf.push(Strip {
                        x: u16::MAX,
                        y: prev_tile.y * Tile::HEIGHT,
                        alpha_idx: alpha_buf.len() as u32,
                        winding: winding_delta,
                    });
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

            strip = Strip {
                x: tile.x * Tile::WIDTH,
                y: tile.y * Tile::HEIGHT,
                alpha_idx: alpha_buf.len() as u32,
                winding: winding_delta,
            };
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
            accumulated_winding = accumulated_winding.madd(sign, h);
            for x_idx in 0..Tile::WIDTH {
                location_winding[x_idx as usize] = location_winding[x_idx as usize].madd(sign, h);
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
            let line_px_left_y = line_top_y
                .madd(px_left_x - line_top_x, y_slope)
                .max_precise(ymin)
                .min_precise(ymax);
            let line_px_right_y = line_top_y
                .madd(px_right_x - line_top_x, y_slope)
                .max_precise(ymin)
                .min_precise(ymax);

            // `x_slope` is always finite, as horizontal geometry is elided.
            let line_px_left_yx =
                f32x4::splat(s, line_top_x).madd(line_px_left_y - line_top_y, x_slope);
            let line_px_right_yx =
                f32x4::splat(s, line_top_x).madd(line_px_right_y - line_top_y, x_slope);
            let h = (line_px_right_y - line_px_left_y).abs();

            // The trapezoidal area enclosed between the line and the right edge of the pixel
            // square.
            let area = 0.5 * h * (2. * px_right_x - line_px_right_yx - line_px_left_yx);
            location_winding[x_idx as usize] =
                location_winding[x_idx as usize] + acc.madd(sign, area);
            acc = acc.madd(sign, h);
        }

        accumulated_winding = accumulated_winding + acc;
    }
}
