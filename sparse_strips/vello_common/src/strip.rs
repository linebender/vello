// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Rendering strips.

use peniko::Fill;

use crate::flatten::Point;
use crate::tile::{Tile, Tiles};

// Note that this will probably disappear and be turned into a const generic in the future.
/// The height of a strip.
pub const STRIP_HEIGHT: usize = Tile::HEIGHT as usize;

/// A strip.
#[derive(Debug, Clone, Copy)]
pub struct Strip {
    /// The x coordinate of the strip, in user coordinates.
    pub x: i32,
    /// The y coordinate of the strip, in user coordinates.
    pub y: u16,
    /// The index into the alpha buffer
    pub col: u32,
    /// The winding number at the start of the strip.
    pub winding: i32,
}

impl Strip {
    /// Return the y coordinate of the strip, in strip units.
    pub fn strip_y(&self) -> u16 {
        self.y / u16::try_from(STRIP_HEIGHT).unwrap()
    }
}

/// Render the tiles stored in `tiles` into the strip and alpha buffer.
/// The strip buffer will be cleared in the beginning.
pub fn render(
    tiles: &Tiles,
    strip_buf: &mut Vec<Strip>,
    alpha_buf: &mut Vec<u32>,
    fill_rule: Fill,
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
    let mut location_winding = [[0_f32; Tile::HEIGHT as usize]; Tile::WIDTH as usize];
    // The accumulated (fractional) windings at this location's right edge. When we move to the
    // next location, this is splatted to that location's starting winding.
    let mut accumulated_winding = [0_f32; Tile::HEIGHT as usize];

    /// A special tile to keep the logic below simple.
    const SENTINEL: Tile = Tile {
        x: i32::MAX,
        y: u16::MAX,
        p0: Point::ZERO,
        p1: Point::ZERO,
    };

    // The strip we're building.
    let mut strip = Strip {
        x: prev_tile.x * Tile::WIDTH as i32,
        y: prev_tile.y * Tile::HEIGHT,
        col: alpha_buf.len() as u32,
        winding: 0,
    };

    for tile in tiles.iter().copied().chain([SENTINEL]) {
        // Push out the winding as an alpha mask when we move to the next location (i.e., a tile
        // without the same location).
        if !prev_tile.same_loc(&tile) {
            macro_rules! fill {
                ($rule:expr) => {
                    for x in 0..Tile::WIDTH as usize {
                        let mut alphas = 0_u32;

                        for y in 0..Tile::HEIGHT as usize {
                            let area = location_winding[x][y];
                            let coverage = $rule(area);
                            let area_u8 = (coverage * 255.0 + 0.5) as u32;

                            alphas += area_u8 << (y * 8);
                        }

                        alpha_buf.push(alphas);
                    }
                };
            }
            match fill_rule {
                Fill::NonZero => {
                    fill!(|area: f32| area.abs().min(1.0))
                }
                Fill::EvenOdd => {
                    // As in other parts of the code, we avoid using `round` since it's very
                    // slow on x86.
                    fill!(|area: f32| (area - 2.0 * ((0.5 * area) + 0.5).floor()).abs())
                }
            };

            #[expect(clippy::needless_range_loop, reason = "dimension clarity")]
            for x in 0..Tile::WIDTH as usize {
                location_winding[x] = accumulated_winding;
            }
        }

        // Push out the strip if we're moving to a next strip.
        if !prev_tile.same_loc(&tile) && !prev_tile.prev_loc(&tile) {
            if !prev_tile.same_row(&tile) {
                winding_delta = 0;
            }

            debug_assert_eq!(
                (prev_tile.x + 1) * Tile::WIDTH as i32 - strip.x,
                alpha_buf.len() as i32 - strip.col as i32,
                "The number of columns written to the alpha buffer should equal the number of columns spanned by this strip."
            );
            strip_buf.push(strip);

            // Once we've reached the `SENTINEL` tile, emit a final strip.
            if tile.y == u16::MAX && tile.x == i32::MAX {
                strip_buf.push(Strip {
                    x: i32::MAX,
                    y: u16::MAX,
                    col: alpha_buf.len() as u32,
                    winding: 0,
                });
                break;
            }

            strip = Strip {
                x: tile.x * Tile::WIDTH as i32,
                y: tile.y * Tile::HEIGHT,
                col: alpha_buf.len() as u32,
                winding: winding_delta,
            };
            // Note: this fill is mathematically not necessary. It provides a way to reduce
            // accumulation of float round errors.
            accumulated_winding.fill(winding_delta as f32);
        }
        prev_tile = tile;

        // TODO: lines are currently still packed into tiles. This will probably change, in which
        // case we will have to translate the lines to have the tile's top-left corner as origin.
        // let line = lines[tile.line_idx as usize];
        let p0_x = tile.p0.x; // - tile_left_x;
        let p0_y = tile.p0.y; // - tile_top_y;
        let p1_x = tile.p1.x; // - tile_left_x;
        let p1_y = tile.p1.y; // - tile_top_y;

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

        let y_slope = (line_bottom_y - line_top_y) / (line_bottom_x - line_top_x);
        let x_slope = 1. / y_slope;

        {
            // The y-coordinate of the intersections between line and the tile's left and right
            // edges respectively.
            //
            // There's some subtety going on here, see the note on `line_px_left_y` below.
            let line_tile_left_y = (line_top_y - line_top_x * y_slope)
                .max(line_top_y)
                .min(line_bottom_y);
            let line_tile_right_y = (line_top_y + (Tile::WIDTH as f32 - line_top_x) * y_slope)
                .max(line_top_y)
                .min(line_bottom_y);

            winding_delta +=
                sign as i32 * ((line_tile_left_y <= 0.) != (line_tile_right_y <= 0.)) as i32;
        }

        for y_idx in 0..Tile::HEIGHT {
            let px_top_y = y_idx as f32;
            let px_bottom_y = 1. + y_idx as f32;

            let ymin = f32::max(line_top_y, px_top_y);
            let ymax = f32::min(line_bottom_y, px_bottom_y);

            let mut acc = 0.;
            for x_idx in 0..Tile::WIDTH {
                let px_left_x = x_idx as f32;
                let px_right_x = 1. + x_idx as f32;

                // The y-coordinate of the intersections between line and the pixel's left and
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
                let line_px_left_y = (line_top_y + (px_left_x - line_top_x) * y_slope)
                    .max(ymin)
                    .min(ymax);
                let line_px_right_y = (line_top_y + (px_right_x - line_top_x) * y_slope)
                    .max(ymin)
                    .min(ymax);

                // `x_slope` is always finite, as horizontal geometry is elided.
                let line_px_left_yx = line_top_x + (line_px_left_y - line_top_y) * x_slope;
                let line_px_right_yx = line_top_x + (line_px_right_y - line_top_y) * x_slope;
                let h = (line_px_right_y - line_px_left_y).abs();

                // The trapezoidal area enclosed between the line and the right edge of the pixel
                // square.
                let area = 0.5 * h * (2. * px_right_x - line_px_right_yx - line_px_left_yx);
                location_winding[x_idx as usize][y_idx as usize] += acc + sign * area;
                acc += sign * h;
            }
            accumulated_winding[y_idx as usize] += acc;
        }
    }
}
