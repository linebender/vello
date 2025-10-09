// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Rendering strips.
use crate::flatten::Line;
use crate::peniko::Fill;
use crate::tile::{Tile, Tiles};
use crate::util::f32_to_u8;
use alloc::vec::Vec;
use bytemuck::{Pod, Zeroable};
use fearless_simd::*;
use std::format;
use std::println;
use std::string::String;
use std::vec;

/// A strip.
#[derive(Debug, Clone, Copy)]
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

    /// Returns the y coordinate of the strip, in strip units.
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

/// Pre-Merge-Tile:
#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct PreMergeTile {
    /// The index into the alpha buffer that an "end tile" should write to.
    pub alpha_index: u32,
    /// Contains tile location information
    pub packed_info: u32,
    /// The exclusive prefix sum of the signed winding number, uploading it skips scan in shader.
    pub scanned_winding: i32,
    /// Necessary padding for alignment
    pub padding: u32,
    /// Line points, adjusted by tile position
    pub p0: [f32; 2],
    /// Line points, adjusted by tile position
    pub p1: [f32; 2],
}

// MSB                                                                  LSB
// 31------------------21|20------------------12|11-------------------3|2|1|0|
// |      Unused (11)    |  Seg Start ID (9)    |  Tile Start ID (9)   |F|T|E|
//
// F = FILL_RULE_MASK
// E = IS_END_TILE_MASK
// T = IS_TILE_FIRST_COL_MASK
const IS_END_TILE_MASK: u32 = 1 << 0;
const IS_TILE_FIRST_COL_MASK: u32 = 1 << 1;
const FILL_RULE_MASK: u32 = 1 << 2;
const TILE_START_ID_SHIFT: u32 = 3;
const SEG_START_ID_SHIFT: u32 = 12;
const ID_MASK: u32 = 0x1ff;
const INVALID_ID: u32 = 256;
const BLOCK_DIM_LG: u32 = 8;

impl PreMergeTile {
    pub fn set_is_end_tile(&mut self, is_end: bool) {
        if is_end {
            self.packed_info |= IS_END_TILE_MASK;
        } else {
            self.packed_info &= !IS_END_TILE_MASK;
        }
    }

    pub fn is_end_tile(&self) -> bool {
        (self.packed_info & IS_END_TILE_MASK) != 0
    }

    pub fn set_is_tile_first_col(&mut self, is_first: bool) {
        if is_first {
            self.packed_info |= IS_TILE_FIRST_COL_MASK;
        }
    }

    pub fn is_tile_first_col(&self) -> bool {
        (self.packed_info & IS_TILE_FIRST_COL_MASK) != 0
    }

    pub fn set_fill_rule(& mut self, is_fill_rule_non_zero: bool) {
        if is_fill_rule_non_zero {
            self.packed_info |= FILL_RULE_MASK;
        }
    }

    pub fn set_tile_start_id(&mut self, id: u32) {
        self.packed_info |= (id & ID_MASK) << TILE_START_ID_SHIFT;
    }

    pub fn tile_start_id(&self) -> u32 {
        (self.packed_info >> TILE_START_ID_SHIFT) & ID_MASK
    }

    pub fn set_seg_start_id(&mut self, id: u32) {
        self.packed_info |= (id & ID_MASK) << SEG_START_ID_SHIFT;
    }

    pub fn seg_start_id(&self) -> u32 {
        (self.packed_info >> SEG_START_ID_SHIFT) & ID_MASK
    }
}

/// Render the tiles stored in `tiles` into the strip and alpha buffer.
/// Aliasing threshold is unused
pub fn render(
    _level: Level,
    tiles: &Tiles,
    strip_buf: &mut Vec<Strip>,      // Output parameter for strips
    pmt_buf: &mut Vec<PreMergeTile>, // Output paramter for pre merge tiles
    alpha_buf: &mut Vec<u8>,         // Output parameter for alpha bytes
    fill_rule: Fill,
    aliasing_threshold: Option<u8>,
    lines: &[Line],
) {
    // let mut winding_fine_ref: Vec<[[f32; 4]; 4]> = Vec::new();
    // let mut winding_acc_ref: Vec<[f32; 4]> = Vec::new();
    // let mut winding_coarse_ref: Vec<i32> = Vec::new();
    // let mut winding_fine_comp: Vec<[[f32; 4]; 4]> = Vec::new();
    // let mut winding_acc_comp: Vec<[f32; 4]> = Vec::new();
    // let mut winding_coarse_comp: Vec<i32> = Vec::new();

    prepare_gpu_inputs(tiles, strip_buf, pmt_buf, alpha_buf, fill_rule, lines);

    // compare_windings(
    //     &winding_fine_ref,
    //     &winding_coarse_ref,
    //     &winding_acc_ref,
    //     &winding_fine_comp,
    //     &winding_coarse_comp,
    //     &winding_acc_comp,
    // );
}

fn render_impl<S: Simd>(
    s: S,
    tiles: &Tiles,
    strip_buf: &mut Vec<Strip>,
    alpha_buf: &mut Vec<u8>,
    fill_rule: Fill,
    aliasing_threshold: Option<u8>,
    lines: &[Line],
    winding_fine_ref: &mut Vec<[[f32; 4]; 4]>,
    winding_acc_ref: &mut Vec<[f32; 4]>,
    winding_coarse_ref: &mut Vec<i32>,
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
    const SENTINEL: Tile = Tile::new(u16::MAX, u16::MAX, 0, false);

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
            //winding_fine_ref.push(location_winding.map(|v| v.val));
            // winding_acc_ref.push(accumulated_winding.val);
            // winding_coarse_ref.push(winding_delta);

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

            //alpha_buf.extend_from_slice(&u8_vals.val);

            #[expect(clippy::needless_range_loop, reason = "dimension clarity")]
            for x in 0..Tile::WIDTH as usize {
                location_winding[x] = accumulated_winding;
            }
        }

        // Push out the strip if we're moving to a next strip.
        if !prev_tile.same_loc(&tile) && !prev_tile.prev_loc(&tile) {
            // debug_assert_eq!(
            //     (prev_tile.x + 1) * Tile::WIDTH - strip.x,
            //     ((alpha_buf.len() - strip.alpha_idx() as usize) / usize::from(Tile::HEIGHT)) as u16,
            //     "The number of columns written to the alpha buffer should equal the number of columns spanned by this strip."
            // );
            //strip_buf.push(strip);

            let is_sentinel = tile_idx == tiles.len() as usize;
            if !prev_tile.same_row(&tile) {
                // Emit a final strip in the row if there is non-zero winding for the sparse fill,
                // or unconditionally if we've reached the sentinel tile to end the path (the
                // `alpha_idx` field is used for width calculations).
                // if winding_delta != 0 || is_sentinel {
                //     strip_buf.push(Strip::new(
                //         u16::MAX,
                //         prev_tile.y * Tile::HEIGHT,
                //         alpha_buf.len() as u32,
                //         should_fill(winding_delta),
                //     ));
                // }

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
            let local_contribution = acc.madd(sign, area);
            if x_idx == 0 {
                //println!("{}: {:<10.4} ", tile_idx, local_contribution.val[0]);
            }
            location_winding[x_idx as usize] = location_winding[x_idx as usize] + local_contribution;
            acc = acc.madd(sign, h);
            location_winding[x_idx as usize] += area.madd(sign, acc);
            acc = h.madd(sign, acc);
        }

        accumulated_winding = accumulated_winding + acc;
    }
}

/// Prepares gpu inputs
fn prepare_gpu_inputs(
    tiles: &Tiles,
    strip_buf: &mut Vec<Strip>,
    pre_merge_buf: &mut Vec<PreMergeTile>,
    alpha_buf: &mut Vec<u8>,
    fill_rule: Fill,
    lines: &[Line],
) {
    if tiles.is_empty() {
        return;
    }

    let should_fill = |winding: i32| match fill_rule {
        Fill::NonZero => winding != 0,
        Fill::EvenOdd => winding % 2 != 0,
    };

    let mut winding_delta: i32 = 0;
    let mut prev_tile = *tiles.get(0);
    let mut alpha_offset: u32 = 0;
    let initial_alpha_len = alpha_buf.len() as u32;

    const SENTINEL: Tile = Tile::new(u16::MAX, u16::MAX, 0, false);

    let mut start_tile_idx = 0;
    let mut start_seg_idx = 0;
    let mut strip = Strip::new(
        prev_tile.x * Tile::WIDTH,
        prev_tile.y * Tile::HEIGHT,
        initial_alpha_len,
        false,
    );
    for (tile_idx, tile) in tiles.iter().copied().chain([SENTINEL]).enumerate() {
        let is_start_tile = tile_idx == 0 || !prev_tile.same_loc(&tile);
        let is_start_segment = tile_idx == 0 || (is_start_tile && !prev_tile.prev_loc(&tile));
        if is_start_tile {
            start_tile_idx = pre_merge_buf.len();
            if tile_idx > 0 {
                alpha_offset += (Tile::WIDTH * Tile::HEIGHT) as u32;
            }
        }

        if is_start_segment {
            start_seg_idx = pre_merge_buf.len();
            strip_buf.push(strip);

            let is_sentinel = tile_idx == tiles.len() as usize;
            if !prev_tile.same_row(&tile) {
                if winding_delta != 0 || is_sentinel {
                    strip_buf.push(Strip::new(
                        u16::MAX,
                        prev_tile.y * Tile::HEIGHT,
                        initial_alpha_len + alpha_offset,
                        should_fill(winding_delta),
                    ));
                }
                winding_delta = 0;
            }

            if is_sentinel {
                break;
            }

            strip = Strip::new(
                tile.x * Tile::WIDTH,
                tile.y * Tile::HEIGHT,
                initial_alpha_len + alpha_offset,
                should_fill(winding_delta),
            );
        }

        let line = lines[tile.line_idx() as usize];
        let tile_left_x = f32::from(tile.x) * f32::from(Tile::WIDTH);
        let tile_top_y = f32::from(tile.y) * f32::from(Tile::HEIGHT);
        let p0_x = line.p0.x - tile_left_x;
        let p0_y = line.p0.y - tile_top_y;
        let p1_x = line.p1.x - tile_left_x;
        let p1_y = line.p1.y - tile_top_y;

        let mut pmt = PreMergeTile {
            alpha_index: initial_alpha_len + alpha_offset,
            packed_info: 0,
            scanned_winding: winding_delta,
            padding: 0,
            p0: [p0_x, p0_y],
            p1: [p1_x, p1_y],
        };

        let sign = (p0_y - p1_y).signum();
        let signed_winding = sign as i32 * tile.winding() as i32;
        winding_delta += signed_winding;

        pmt.set_is_end_tile(
            tile_idx == tiles.len() as usize - 1 ||
            !tiles.get(tile_idx as u32 + 1).same_loc(&tile),
        );
        pmt.set_is_tile_first_col(tile.x == 0);

        let block_idx = pre_merge_buf.len() >> BLOCK_DIM_LG << BLOCK_DIM_LG;
        let start_tile_pack = if start_tile_idx < block_idx {
            INVALID_ID
        } else {
            (start_tile_idx  - block_idx) as u32
        };
        pmt.set_tile_start_id(start_tile_pack);

        let start_seg_pack = if start_seg_idx < block_idx {
            INVALID_ID
        } else {
            (start_seg_idx - block_idx) as u32
        };
        pmt.set_seg_start_id(start_seg_pack);
        pmt.set_fill_rule(fill_rule == Fill::NonZero);

        pre_merge_buf.push(pmt);
        prev_tile = tile;
    }

    //TODO alpha buff should be removed in place of a single accumulated u32
    alpha_buf.resize(alpha_buf.len() + alpha_offset as usize, 0);
}


#[derive(Debug, Clone, Copy, Default)]
pub struct StitchIndicator {
    /// True if the tile's location dependency must be stitched.
    pub loc_stitch_required: bool,
    /// True if the tile's accumulation dependency must be stitched.
    pub acc_stitch_required: bool,
    /// The ID of the partition (workgroup) that processed this tile.
    pub partition_id: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PartitionIndicator {
    /// Does this partition contain a `seg_start`?
    pub has_seg_start: bool,
    /// Does this partition contain a `tile_start`?
    pub has_tile_start: bool,
}

/// Simulate the merge_shader using the PreMergeTile intermediate representation.
/// Note: THIS WILL NOT WORK with the current prepare_gpu_inputs because it is expecting local
/// indexes relative to the call of strip::render, rather than the current global indexes provided
/// by using the pre_merge_buffer length.
fn cpu_merge(
    pmt_buf: &mut Vec<PreMergeTile>,
    alpha_buf: &mut Vec<u8>,
    fill_rule: Fill,
    winding_fine_comp: &mut Vec<[[f32; 4]; 4]>,
    winding_acc_comp: &mut Vec<[f32; 4]>,
    winding_coarse_comp: &mut Vec<i32>,
) {
    if pmt_buf.is_empty() {
        return;
    }

    let pmt_count = pmt_buf.len();
    let mut temp_acc = vec![[0f32; Tile::HEIGHT as usize]; pmt_count];
    let mut temp_fine = vec![[[0f32; Tile::HEIGHT as usize]; Tile::WIDTH as usize]; pmt_count];

    for gid in 0..pmt_buf.len() {
        let pmt = pmt_buf[gid];
        let p0_x = pmt.p0[0];
        let p0_y = pmt.p0[1];
        let p1_x = pmt.p1[0];
        let p1_y = pmt.p1[1];

        if (p0_y - p1_y).abs() < 1e-6 {
            continue;
        }

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


        let dx = line_bottom_x - line_top_x; // this becomes the operative issue.
        let is_vertical = dx.abs() < 1e-6;  // Now use an epsilon to be sure
        let dy = line_bottom_y - line_top_y; // we skip horizontal or close to horizontal lines by now
        let y_slope = if dx == 0.0 { f32::MAX } else { dy / dx };
        let x_slope = if dy == 0.0 { f32::MAX } else { dx / dy };
        let sign = (p0_y - p1_y).signum();

        if pmt.is_tile_first_col() && line_left_x < 0. {
            let (ymin, ymax) = if is_vertical {
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

            let px_top_y_arr = [0.0, 1.0, 2.0, 3.0];
            let mut h = [0.0; Tile::HEIGHT as usize];
            for y in 0..Tile::HEIGHT as usize {
                let px_top_y: f32 = px_top_y_arr[y];
                let px_bottom_y: f32 = 1.0 + px_top_y;
                let ymin_clamped = px_top_y.max(ymin);
                let ymax_clamped = px_bottom_y.min(ymax);
                h[y] = (ymax_clamped - ymin_clamped).max(0.0);
                temp_acc[gid][y] = sign * h[y];
            }

            if line_right_x < 0. {
                for x_idx in 0..Tile::WIDTH as usize {
                    for y in 0..Tile::HEIGHT as usize {
                        temp_fine[gid][x_idx][y] = temp_acc[gid][y];
                    }
                }
                continue;
            }
        }

        if is_vertical {
            let line_x = line_top_x;
            let event_x_idx = line_x.floor() as usize;
            for x_idx in 0..Tile::WIDTH as usize {
                for y in 0..Tile::HEIGHT as usize {
                    let mut h = 0.0;
                    let mut area = 0.0;
                    if x_idx == event_x_idx {
                        let px_top_y = y as f32;
                        let px_bottom_y = 1.0 + px_top_y;

                        let ymin = line_top_y.max(px_top_y);
                        let ymax = line_bottom_y.min(px_bottom_y);
                        let coverage_right = (x_idx as f32 + 1.0) - line_x;
                        h = (ymax - ymin).max(0.0);
                        area = h * coverage_right;
                    }
                    temp_fine[gid][x_idx][y] = temp_acc[gid][y] + sign * area;
                    temp_acc[gid][y] += sign * h;
                }
            }
        } else {
            for x_idx in 0..Tile::WIDTH {
                let x_idx_f = x_idx as f32;
                let px_left_x = x_idx_f;
                let px_right_x = 1.0 + x_idx_f;

                for y in 0..Tile::HEIGHT as usize {
                    let px_top_y = y as f32;
                    let px_bottom_y = 1.0 + px_top_y;
                    let ymin = line_top_y.max(px_top_y);
                    let ymax = line_bottom_y.min(px_bottom_y);

                    let line_px_left_y = (line_top_y + (px_left_x - line_top_x) * y_slope)
                        .max(ymin)
                        .min(ymax);
                    let line_px_right_y = (line_top_y + (px_right_x - line_top_x) * y_slope)
                        .max(ymin)
                        .min(ymax);

                    let line_px_left_yx = line_top_x + (line_px_left_y - line_top_y) * x_slope;
                    let line_px_right_yx = line_top_x + (line_px_right_y - line_top_y) * x_slope;
                    let h = (line_px_right_y - line_px_left_y).abs();
                    let area = 0.5 * h * (2. * px_right_x - line_px_right_yx - line_px_left_yx);

                    temp_fine[gid][x_idx as usize][y] = temp_acc[gid][y] + sign * area;
                    temp_acc[gid][y] += sign * h;
                }
            }
        }
    }

    const BLOCK_DIM: usize = 256;
    let GRID_DIM: usize = (pmt_buf.len() + BLOCK_DIM - 1) / BLOCK_DIM;
    let mut stitch_indicator: Vec<StitchIndicator> = Vec::new();
    let mut part_indicator: Vec<PartitionIndicator> = Vec::new();
    let mut stitch_loc: Vec<[[f32; Tile::HEIGHT as usize]; Tile::WIDTH as usize]> = Vec::new();
    let mut part_loc: Vec<[[f32; Tile::HEIGHT as usize]; Tile::WIDTH as usize]> = Vec::new();
    let mut part_acc_seg: Vec<[f32; Tile::HEIGHT as usize]> = Vec::new();
    let mut part_acc_mine: Vec<[f32; Tile::HEIGHT as usize]> = Vec::new();

    for wgid in 0..GRID_DIM {
        let mut wg_loc = [[[0f32; Tile::HEIGHT as usize]; Tile::WIDTH as usize]; BLOCK_DIM];
        let mut wg_acc = [[0f32; Tile::HEIGHT as usize]; BLOCK_DIM];

        // dirty, blend the scan with the stitch preparation
        for tid in 0..BLOCK_DIM {
            let gid = tid + wgid * BLOCK_DIM;
            if gid < pmt_count {
                wg_acc[tid] = temp_acc[gid];
                wg_loc[tid] = temp_fine[gid];
            }

            if tid > 0 {
                for y in 0..Tile::HEIGHT as usize {
                    wg_acc[tid][y] += wg_acc[tid - 1][y];
                }
                for x in 0..Tile::WIDTH as usize {
                    for y in 0..Tile::HEIGHT as usize {
                        wg_loc[tid][x][y] += wg_loc[tid - 1][x][y];
                    }
                }
            }
            // barrier

            // now duplicate previous logic. . .
            if gid < pmt_count {
                let pmt = pmt_buf[gid];
                if pmt.is_end_tile() || tid == BLOCK_DIM - 1 {
                    let mut my_acc = wg_acc[tid];
                    let mut my_loc = wg_loc[tid];

                    let tile_start_id = pmt.tile_start_id() as usize;
                    let seg_start_id = pmt.seg_start_id() as usize;
                    let tile_start_valid = tile_start_id != INVALID_ID as usize;
                    let seg_start_valid = seg_start_id != INVALID_ID as usize;

                    if tile_start_valid {
                        if tile_start_id != 0 {
                            for x in 0..Tile::WIDTH as usize {
                                for y in 0..Tile::HEIGHT as usize {
                                    my_loc[x][y] -= wg_loc[tile_start_id - 1][x][y];
                                }
                            }

                            my_acc = wg_acc[tile_start_id - 1];
                        } else {
                            my_acc = [0f32; 4];
                        }
                    }

                    if seg_start_valid {
                        let scanned_winding = pmt_buf[wgid * BLOCK_DIM + seg_start_id].scanned_winding as f32;
                        if seg_start_id != 0 {
                            for y in 0..Tile::HEIGHT as usize {
                                my_acc[y] += scanned_winding - wg_acc[seg_start_id - 1][y];
                            }
                        } else {
                            for y in 0..Tile::HEIGHT as usize {
                                my_acc[y] += scanned_winding;
                            }
                        }
                    }

                    if tile_start_valid {
                        for x in 0..Tile::WIDTH as usize {
                            for y in 0..Tile::HEIGHT as usize {
                                my_loc[x][y] += my_acc[y];
                            }
                        }
                    }

                    let mut stitch = StitchIndicator {
                        loc_stitch_required: false,
                        acc_stitch_required: false,
                        partition_id: 0,
                    };

                    // only end tiles participate in the write out
                    if pmt.is_end_tile() {
                        if tile_start_valid {
                            if seg_start_valid {
                                temp_fine[gid] = my_loc;
                            } else {
                                stitch.acc_stitch_required = true;
                                stitch.partition_id = wgid;
                            }
                        } else {
                            stitch.loc_stitch_required = true;
                            stitch.partition_id = wgid;
                        }

                    }
                    stitch_loc.push(my_loc);
                    stitch_indicator.push(stitch);

                    if tid == BLOCK_DIM - 1 {
                        // Full OR tile - seg_start
                        part_acc_mine.push(my_acc);

                        // Push Full
                        let mut seg_acc = [0f32; Tile::HEIGHT as usize];
                        if seg_start_valid {
                            let scanned_winding = pmt_buf[wgid * BLOCK_DIM + seg_start_id].scanned_winding as f32;
                            if seg_start_id != 0 {
                                for y in 0..Tile::HEIGHT as usize {
                                    seg_acc[y] = scanned_winding + wg_acc[tid][y] -
                                        wg_acc[seg_start_id - 1][y];
                                }
                            } else {
                                for y in 0..Tile::HEIGHT as usize {
                                    seg_acc[y] = scanned_winding + wg_acc[tid][y];
                                }
                            }
                            part_acc_seg.push(seg_acc);
                        } else {
                            part_acc_seg.push(wg_acc[tid]);
                        }

                        if tile_start_valid && tile_start_id != 0 {
                            let mut p_loc = [[0f32; Tile::HEIGHT as usize]; Tile::WIDTH as usize];
                            for x in 0..Tile::WIDTH as usize {
                                for y in 0..Tile::HEIGHT as usize {
                                    p_loc[x][y] += wg_loc[tid][x][y] - wg_loc[pmt.tile_start_id() as usize - 1][x][y];
                                }
                            }
                            part_loc.push(p_loc);
                        } else {
                            part_loc.push(wg_loc[tid]);
                        }
                        let p_ind = PartitionIndicator {
                            has_seg_start: seg_start_valid,
                            has_tile_start: tile_start_valid,
                        };
                        part_indicator.push(p_ind);
                    }
                } else {
                    let stitch = StitchIndicator {
                        loc_stitch_required: false,
                        acc_stitch_required: false,
                        partition_id: 0,
                    };
                    stitch_loc.push([[0f32; 4]; 4]);
                    stitch_indicator.push(stitch);
                }
            }
        }
    }

    //Stitching is not done in grid blocked
    for gid in 0..pmt_count {
        let s_ind = stitch_indicator[gid];
        if s_ind.loc_stitch_required && !s_ind.acc_stitch_required {
            let mut loc = stitch_loc[gid];
            let mut lookback_id = s_ind.partition_id - 1;
            let mut part_ind:PartitionIndicator;
            loop {
                // Sum the prev
                for x in 0..Tile::WIDTH as usize {
                    for y in 0..Tile::HEIGHT as usize {
                        loc[x][y] += part_loc[lookback_id][x][y];
                    }
                }

                // Did we hit a tile_start?
                part_ind = part_indicator[lookback_id];
                if part_ind.has_tile_start {
                    break;
                } else {
                    lookback_id -= 1;
                }
            }

            //Once we hit a tile start. . . Then we have to stitch the acc
            //Since this partition is guaranteed to have the tile start
            let mut acc = part_acc_mine[lookback_id];

            // If this partition also had a seg start, we're done, otherwise we need to continue
            // the traversal
            if !part_ind.has_seg_start {
                lookback_id -= 1;
                loop {
                    // Sum the prev
                    for y in 0..Tile::HEIGHT as usize {
                        acc[y] += part_acc_seg[lookback_id][y];
                    }

                    // Did we hit a seg_start?
                    part_ind = part_indicator[lookback_id];
                    if part_ind.has_seg_start {
                        break;
                    } else {
                        lookback_id -= 1;
                    }
                }
            }

            //Push out
            temp_acc[gid] = acc;
            for x in 0..Tile::WIDTH as usize {
                for y in 0..Tile::HEIGHT as usize {
                    temp_fine[gid][x][y] = loc[x][y] + acc[y];
                }
            }
        }

        if s_ind.acc_stitch_required && !s_ind.loc_stitch_required {
            let mut acc = [0f32; Tile::HEIGHT as usize];
            let mut lookback_id = s_ind.partition_id - 1;
            loop {
                let s_acc = part_acc_seg[lookback_id];
                for y in 0..Tile::HEIGHT as usize {
                    acc[y] += s_acc[y];
                }
                if part_indicator[lookback_id].has_seg_start {
                    break;
                } else {
                    lookback_id -= 1;
                }
            }

            // Push out
            let loc = stitch_loc[gid];
            for x in 0..Tile::WIDTH as usize {
                for y in 0..Tile::HEIGHT as usize {
                    temp_fine[gid][x][y] = loc[x][y] + acc[y];
                }
            }
        }
    }

    for gid in 0..(pmt_count - 1) {
        let pmt = pmt_buf[gid];
        if pmt.is_end_tile() {
            winding_fine_comp.push(temp_fine[gid]);
            winding_acc_comp.push(temp_acc[gid + 1]);
            winding_coarse_comp.push(pmt_buf[gid + 1].scanned_winding);

            match fill_rule {
                Fill::NonZero => {
                    for x in 0..Tile::WIDTH as usize {
                        for y in 0..Tile::HEIGHT as usize {
                            let area = temp_fine[gid][x][y];
                            let coverage = area.abs();
                            let mulled = 0.5 + coverage * 255.0;
                            temp_fine[gid][x][y] = mulled.min(255.0);
                        }
                    }
                }
                Fill::EvenOdd => {
                    for x in 0..Tile::WIDTH as usize {
                        for y in 0..Tile::HEIGHT as usize {
                            let area = temp_fine[gid][x][y];
                            let im1 = (0.5 + area * 0.5).floor();
                            let coverage = (im1 + area * -2.0).abs();
                            let mulled = 0.5 + coverage * 255.0;
                            temp_fine[gid][x][y] = mulled.min(255.0);
                        }
                    }
                }
            };

            let mut u8_vals = [0u8; 16];
            let mut i = 0;
            for x in 0..Tile::WIDTH as usize {
                for y in 0..Tile::HEIGHT as usize {
                    u8_vals[i] = temp_fine[gid][x][y].round() as u8;
                    i += 1;
                }
            }

            // this will be indexed into
            alpha_buf.extend_from_slice(&u8_vals);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
pub fn compare_windings(
    winding_fine_ref: &[[[f32; 4]; 4]],
    winding_coarse_ref: &[i32],
    winding_acc_ref: &[[f32; 4]],
    winding_fine_comp: &[[[f32; 4]; 4]],
    winding_coarse_comp: &[i32],
    winding_acc_comp: &[[f32; 4]],
) -> bool {
    let coarse_match = compare_coarse_windings(winding_coarse_ref, winding_coarse_comp);
    if !coarse_match {
        println!("\nComparison halted due to coarse winding mismatch.");
        return false;
    }
    let fine_match = compare_fine_windings(winding_fine_ref, winding_fine_comp);
    if !fine_match {
        println!("\nComparison failed at fine winding stage.");
        return false;
    }
    let acc_match = compare_acc_windings(winding_acc_ref, winding_acc_comp);

    acc_match && coarse_match && fine_match
}

fn compare_coarse_windings(reference: &[i32], comp: &[i32]) -> bool {
    if reference.len() != comp.len() {
        println!(
            "❌ FATAL ERROR: Coarse winding vectors have different lengths! Reference: {}, Comp: {}",
            reference.len(),
            comp.len()
        );
        return false;
    }
    let mut mismatches = 0;
    for (i, (ref_val, comp_val)) in reference.iter().zip(comp.iter()).enumerate() {
        if ref_val != comp_val {
            if mismatches == 0 {
                println!("\n--- Coarse Winding Mismatches Found ---");
            }
            println!(
                "Mismatch at tile index {}: Reference = {}, Comp = {}",
                i, ref_val, comp_val
            );
            mismatches += 1;
        }
    }
    if mismatches == 0 {
        true
    } else {
        println!(
            "❌ Coarse winding comparison FAILED. Found {} mismatches out of {} tiles.",
            mismatches,
            reference.len()
        );
        false
    }
}

fn compare_fine_windings(reference: &[[[f32; 4]; 4]], comp: &[[[f32; 4]; 4]]) -> bool {
    if reference.len() != comp.len() {
        println!(
            "❌ FATAL ERROR: Fine winding vectors have different lengths! Reference: {}, Comp: {}",
            reference.len(),
            comp.len()
        );
        return false;
    }

    let mut total_mismatches = 0;
    let mut first_mismatch_tile_index = None;

    for i in 0..reference.len() {
        let ref_tile = reference[i];
        let comp_tile = comp[i];
        let mut tile_has_mismatch = false;
        for x in 0..4 {
            for y in 0..4 {
                let ref_val = ref_tile[x][y];
                let comp_val = comp_tile[x][y];
                if (ref_val - comp_val).abs() > 0.01 {
                    total_mismatches += 1;
                    tile_has_mismatch = true;
                }
            }
        }
        if tile_has_mismatch && first_mismatch_tile_index.is_none() {
            first_mismatch_tile_index = Some(i);
        }
    }

    if total_mismatches == 0 {
        true
    } else {
        println!(
            "❌ Fine winding comparison FAILED. Found {} total mismatches.",
            total_mismatches
        );
        if let Some(i) = first_mismatch_tile_index {
            println!(
                "\n--- Detailed Mismatch Report for First Failing Tile (Index {}) ---",
                i
            );
            println!("Column-major format: pixel (x, y)");
            println!("{:<45} {:<45}", "Reference (cpu_merge_ref)", "Comp");
            println!("{:-<45} {:-<45}", "-", "-");

            let ref_tile = reference[i];
            let comp_tile = comp[i];

            for y in 0..4 {
                let mut ref_row_str = String::new();
                let mut comp_row_str = String::new();
                for x in 0..4 {
                    ref_row_str.push_str(&format!("({x},{y})={:<+8.4} ", ref_tile[x][y]));
                    comp_row_str.push_str(&format!("({x},{y})={:<+8.4} ", comp_tile[x][y]));
                }
                println!("{}   {}", ref_row_str.trim_end(), comp_row_str.trim_end());
            }
        }
        false
    }
}

fn compare_acc_windings(reference: &[[f32; 4]], comp: &[[f32; 4]]) -> bool {
    if reference.len() != comp.len() {
        println!(
            "❌ FATAL ERROR: Accumulator winding vectors have different lengths! Reference: {}, Comp: {}",
            reference.len(),
            comp.len()
        );
        return false;
    }
    let mut mismatches = 0;
    for i in 0..reference.len() {
        let ref_tile_acc = reference[i];
        let comp_tile_acc = comp[i];
        for y in 0..4 {
            if (ref_tile_acc[y] - comp_tile_acc[y]).abs() > 0.01 {
                if mismatches == 0 {
                    println!("\n--- Accumulator Winding Mismatches Found ---");
                }
                println!(
                    "Mismatch at tile index {} row {}: Reference = {:.4}, Comp = {:.4}",
                    i, y, ref_tile_acc[y], comp_tile_acc[y]
                );
                mismatches += 1;
            }
        }
    }
    if mismatches == 0 {
        true
    } else {
        println!(
            "❌ Accumulator winding comparison FAILED. Found {} mismatches.",
            mismatches,
        );
        false
    }
}
