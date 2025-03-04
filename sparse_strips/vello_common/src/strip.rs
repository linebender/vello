// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Rendering strips.

use crate::footprint::Footprint;
use crate::tile::Tiles;
use peniko::Fill;

// Note that this will probably disappear and be turned into a const generic in the future.
/// The height of a strip.
pub const STRIP_HEIGHT: usize = 4;

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

    let mut strip_start = true;
    let mut cols = alpha_buf.len() as u32;
    let mut prev_tile = tiles.get(0);
    let mut fp = prev_tile.footprint();
    let mut seg_start = 0;
    let mut delta = 0;

    // Note: the input should contain a sentinel tile, to avoid having
    // logic here to process the final strip.
    for i in 1..tiles.len() {
        let cur_tile = tiles.get(i);

        if !prev_tile.same_loc(cur_tile) {
            let start_delta = delta;
            let same_strip = prev_tile.prev_loc(cur_tile);

            if same_strip {
                fp.extend(3);
            }

            let x0 = fp.x0();
            let x1 = fp.x1();
            let mut areas = [[start_delta as f32; 4]; 4];

            for j in seg_start..i {
                let tile = tiles.get(j);

                delta += tile.delta();

                let p0 = tile.p0;
                let p1 = tile.p1;
                let inv_slope = (p1.x - p0.x) / (p1.y - p0.y);

                // Note: We are iterating in column-major order because the inner loop always
                // has a constant number of iterations, which makes it more SIMD-friendly. Worth
                // running some tests whether a different order allows for better performance.
                for x in x0..x1 {
                    // Relative x offset of the start point from the
                    // current column.
                    let rel_x = p0.x - x as f32;

                    for y in 0..4 {
                        // Relative y offset of the start
                        // point from the current row.
                        let rel_y = p0.y - y as f32;
                        // y values will be 1 if the point is below the current row,
                        // 0 if the point is above the current row, and between 0-1
                        // if it is on the same row.
                        let y0 = rel_y.clamp(0.0, 1.0);
                        let y1 = (p1.y - y as f32).clamp(0.0, 1.0);
                        // If != 0, then the line intersects the current row
                        // in the current tile.
                        let dy = y0 - y1;

                        // x intersection points in the current tile.
                        let xx0 = rel_x + (y0 - rel_y) * inv_slope;
                        let xx1 = rel_x + (y1 - rel_y) * inv_slope;
                        let xmin0 = xx0.min(xx1);
                        let xmax = xx0.max(xx1);
                        // Subtract a small delta to prevent a division by zero below.
                        let xmin = xmin0.min(1.0) - 1e-6;
                        // Clip x_max to the right side of the pixel.
                        let b = xmax.min(1.0);
                        // Clip x_max to the left side of the pixel.
                        let c = b.max(0.0);
                        // Clip x_min to the left side of the pixel.
                        let d = xmin.max(0.0);
                        // Calculate the covered area.
                        // TODO: How is this formula derived?
                        let mut a = (b + 0.5 * (d * d - c * c) - xmin) / (xmax - xmin);
                        // a can be NaN if dy == 0 (and thus xmax - xmin = 0, resulting in
                        // a division by 0 above). This code changes those NaNs to 0.
                        a = a.abs().max(0.).copysign(a);

                        areas[x as usize][y] += a * dy;

                        // Making this branchless doesn't lead to any performance improvements
                        // according to my measurements.
                        if p0.x == 0.0 {
                            areas[x as usize][y] += (y as f32 - p0.y + 1.0).clamp(0.0, 1.0);
                        } else if p1.x == 0.0 {
                            areas[x as usize][y] -= (y as f32 - p1.y + 1.0).clamp(0.0, 1.0);
                        }
                    }
                }
            }

            macro_rules! fill {
                ($rule:expr) => {
                    for x in x0..x1 {
                        let mut alphas = 0_u32;

                        for y in 0..4 {
                            let area = areas[x as usize][y];
                            let area_u8 = $rule(area);

                            alphas += area_u8 << (y * 8);
                        }

                        alpha_buf.push(alphas);
                    }
                };
            }

            match fill_rule {
                Fill::NonZero => {
                    fill!(|area: f32| (area.abs().min(1.0) * 255.0 + 0.5) as u32)
                }
                Fill::EvenOdd => {
                    fill!(|area: f32| {
                        let area_abs = area.abs();
                        let area_fract = area_abs.fract();
                        let odd = area_abs as i32 & 1;
                        // Even case: 2.68 -> The opacity should be (0 + 0.68) = 68%.
                        // Odd case: 1.68 -> The opacity should be (1 - 0.68) = 32%.
                        // `add_val` represents the 1, sign represents the minus.
                        // If we have for example 2.68, then opacity is 68%, while for
                        // 1.68 it would be (1 - 0.68) = 32%.
                        // So for odd, add_val should be 1, while for even it should be 0.
                        let add_val = odd as f32;
                        // 1 for even, -1 for odd.
                        let sign = -2.0 * add_val + 1.0;
                        let factor = add_val + sign * area_fract;

                        (factor * 255.0 + 0.5) as u32
                    })
                }
            }

            if strip_start {
                let strip = Strip {
                    x: 4 * prev_tile.x + x0 as i32,
                    y: 4 * prev_tile.y,
                    col: cols,
                    winding: start_delta,
                };

                strip_buf.push(strip);
            }

            cols += x1 - x0;
            fp = if same_strip {
                Footprint::from_index(0)
            } else {
                Footprint::empty()
            };

            strip_start = !same_strip;
            seg_start = i;

            if !prev_tile.same_row(cur_tile) {
                delta = 0;
            }
        }

        fp.merge(&cur_tile.footprint());

        prev_tile = cur_tile;
    }
}
