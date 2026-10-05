// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Port of Skia's `StripProcessorSimd.h`, adapted to 16 samples per pixel and 16×16 tiles.
//! `resolveWindingToAlpha` becomes a resolve to sample masks (see the module docs in `mod.rs`).
//!
//! Skia packs the 8 per-sample windings of a pixel into a `skvx::Vec<2, uint32_t>`; here the 16
//! per-sample windings of a pixel are 16 byte lanes of a [`SwarPixel`] (see [`swar`]). Where the
//! SIMD version hard-codes 8 samples (truncation masks, the mask-to-lanes expansion, the alpha
//! resolve), the 16 sample equivalents follow `StripProcessorScalar.h`, which is generic over the
//! sample count.
//!
//! The core function is [`StripProcessor::rasterize_line_to_tile`]. It takes the line, coarse
//! winding, and intersection mask and converts it to subsample winding. There are two core
//! techniques that make MSAA calculation on the CPU feasible:
//!
//! 1. LUT lookup (sub-pixel coverage): evaluating the line equation at 16 subsample locations per
//!    pixel is prohibitively expensive on the CPU. Instead, a fixed set of subsample coverage
//!    patterns are precomputed into a look-up table (see [`super::lut`]).
//! 2. Hierarchical winding: a naive scanline renderer would carry 16 scanlines per pixel row across
//!    the tiles. Instead, the "coarse winding" is only carried at the top left corner of the tile,
//!    and each subsample reconstructs its winding from that point using the intersected lines.
//!
//! Conceptually, rasterizing a line into a tile has a "coarse phase", which calculates the aliased
//! winding of the pixels not intersected by the line, and a "fine phase" for the pixels it
//! intersects:
//!
//! - Coarse winding: when processing begins for a new spatial tile, the coarse winding at its top
//!   left corner seeds the subsample winding. Tiles rasterized at the same spatial location
//!   accumulate the net winding at the top right corner of the tile, which becomes the top left
//!   backdrop of the next tile in the row.
//! - Crossing top and propagating right: crossing the top edge of a pixel toggles the winding of
//!   all pixels to the right of the crossing pixel.
//! - Fill left: lines don't necessarily span the entire height of the tile, so if a line crosses
//!   the left boundary of the tile, all pixel rows below that crossing are toggled.
//! - Per-row intersections: the intersection of the line with each pixel row edge is found, using
//!   the tile-edge intersections from [`Tile::clip_to_tile`] as the first and last entry.
//! - LUT lookup: the LUT is indexed by a normalized slope and a translation `t`. `t` is stepped
//!   incrementally (DDA) across the pixels, see [`StripProcessor::compute_line_step_params`].
//! - Truncation: the LUT always projects the line across the full height of the pixel. Where a line
//!   ends inside a pixel (or on a right pixel edge), the samples above the start or below the end
//!   are masked out, since the sibling line sharing the endpoint (paths are watertight) accounts
//!   for them. See [`StripProcessor::get_truncation_mask`].
//! - Coverage inversion: the LUT mask always marks the samples right of the line. For the nonzero
//!   rule the contribution of a pixel is `±(LUT - invert)`, where the sign is the y direction of
//!   the line and `invert` moves the winding below the line for lines that don't cross the top of
//!   the pixel row. For even-odd, the contribution is `LUT ^ invert`.
//!
//! `StripProcessorScalar.h` has the full discussion, with diagrams, of the truncation and inversion
//! rules and their interaction with left edge touches.

use super::lut::{
    LUT_MASK_HEIGHT, LUT_MASK_WIDTH, LUT_MASK_WIDTH_EXCL, LutArray, NUM_SUB_SAMPLES, SubSampleType,
};
use super::swar::{self, SwarPixel};
use super::tile::{Line, Point, Tile};
use super::{MASK_WORDS_PER_TILE, TILE_SIZE};

const TILE_WIDTH: usize = TILE_SIZE as usize;
const TILE_HEIGHT: usize = TILE_SIZE as usize;

/// Skia's `SparseStripConfig::kStripEpsilon`.
pub(super) const STRIP_EPSILON: f32 = 1e-5;

/// All samples (Skia's `0xff` for 8 samples).
const FULL_MASK: SubSampleType = SubSampleType::MAX;

/// Per-line parameters for stepping through the LUT. See
/// [`StripProcessor::compute_line_step_params`].
struct LineStepParams<'a> {
    mask_row_lut: &'a [SubSampleType; LUT_MASK_WIDTH],
    step_x_fixed: i32,
    step_y_fixed: i32,
    t_base_fixed: i32,
    sorted_x_dir: bool,
}

/// Accumulates the per-sample winding of one spatial tile and resolves it to sample masks.
///
/// `IS_WINDING` selects the nonzero (`true`) or even-odd (`false`) fill rule.
pub(super) struct StripProcessor<'a, const IS_WINDING: bool> {
    /// The winding of each sample of each pixel of the tile, `[row][column]`.
    ///
    /// For nonzero, a lane holds `0x80 + winding` (wrapping), so that `0x80` is winding 0. For
    /// even-odd, a lane holds the parity of the winding (0 or 1).
    subsample_winding: [[SwarPixel; TILE_WIDTH]; TILE_HEIGHT],
    /// Integer winding at the top left corner of the tile.
    coarse_winding: i32,
    /// The lines of the path, indexed by [`Tile::line_idx`].
    lines: &'a [Line],
    /// The slope intercept lookup table used to evaluate subsample winding.
    mask_lut: &'a LutArray,
}

impl<'a, const IS_WINDING: bool> StripProcessor<'a, IS_WINDING> {
    /// Skia's `kInitialWinding`: the lane value of winding 0.
    const INITIAL_WINDING: u8 = if IS_WINDING { 0x80 } else { 0 };

    pub(super) fn new(lines: &'a [Line], mask_lut: &'a LutArray) -> Self {
        let mut processor = Self {
            subsample_winding: [[0; TILE_WIDTH]; TILE_HEIGHT],
            coarse_winding: 0,
            lines,
            mask_lut,
        };
        processor.clear_winding(Self::INITIAL_WINDING);
        processor
    }

    #[inline(always)]
    fn clear_winding(&mut self, val: u8) {
        self.subsample_winding = [[swar::splat(val); TILE_WIDTH]; TILE_HEIGHT];
    }

    /// Seed every sample of the tile with the coarse winding.
    #[inline(always)]
    pub(super) fn clear_with_coarse_winding(&mut self) {
        let winding_byte = if IS_WINDING {
            // Cast to i8 to sign extend before casting to u8, then apply SWAR bias to map
            // -127/128 to 0/255.
            0x80_u8.wrapping_add(self.coarse_winding as i8 as u8)
        } else {
            u8::from(self.coarse_winding & 1 != 0)
        };
        self.clear_winding(winding_byte);
    }

    /// Whether winding `w` is inside per the fill rule (Skia's static `ShouldFill`).
    #[inline(always)]
    pub(super) fn should_fill(&self, w: i32) -> bool {
        if IS_WINDING { w != 0 } else { (w & 1) != 0 }
    }

    #[inline(always)]
    pub(super) fn coarse_winding(&self) -> i32 {
        self.coarse_winding
    }

    #[inline(always)]
    pub(super) fn set_coarse_winding(&mut self, val: i32) {
        self.coarse_winding = val;
    }

    /// Port of `resolveWindingToAlpha`: resolve the subsample winding of the tile to sample masks.
    ///
    /// Writes all of `dst`, in the layout of [`super::StripSink::scratch_mut`]: pixel `(x, y)` is
    /// the `u16` at index `y * 16 + x`, and bit `k` is set when sample `k` is inside per the fill
    /// rule.
    pub(super) fn resolve_masks(&self, dst: &mut [u32; MASK_WORDS_PER_TILE]) {
        for (row, dst_row) in self
            .subsample_winding
            .iter()
            .zip(dst.chunks_exact_mut(TILE_WIDTH / 2))
        {
            for (pair, word) in row.chunks_exact(2).zip(dst_row) {
                let lo = Self::get_actives_wide(pair[0]);
                let hi = Self::get_actives_wide(pair[1]);
                *word = u32::from(lo) | (u32::from(hi) << 16);
            }
        }
    }

    /// Accumulate the winding of the line of `tile` into the tile at `tile_bounds` (the top left
    /// and bottom right corners of the tile in device space).
    pub(super) fn rasterize_line_to_tile(&mut self, tile: Tile, tile_bounds: [Point; 2]) {
        let line = self.lines[tile.line_idx() as usize];
        let canonical_y_dir = line.p1.y >= line.p0.y;
        if canonical_y_dir {
            self.rasterize_line_to_tile_impl::<true>(tile, tile_bounds, line);
        } else {
            self.rasterize_line_to_tile_impl::<false>(tile, tile_bounds, line);
        }
    }

    fn rasterize_line_to_tile_impl<const CANONICAL_Y_DIR: bool>(
        &mut self,
        tile: Tile,
        tile_bounds: [Point; 2],
        line: Line,
    ) {
        let canonical_x_dir = line.p1.x >= line.p0.x;

        // Accumulate the coarse winding for the next spatial tile; `subsample_winding` has already
        // been seeded during the tile state transition.
        let winding_bit = i32::from(tile.coarse_winding());
        if IS_WINDING {
            self.coarse_winding += if CANONICAL_Y_DIR {
                winding_bit
            } else {
                -winding_bit
            };
        } else {
            self.coarse_winding ^= winding_bit;
        }

        let dx = line.p1.x - line.p0.x;
        let dy = line.p1.y - line.p0.y;
        let inv_dx = if dx.abs() <= STRIP_EPSILON {
            0.0
        } else {
            1.0 / dx
        };
        let inv_dy = if dy.abs() <= STRIP_EPSILON {
            0.0
        } else {
            1.0 / dy
        };
        let dxdy = dx * inv_dy;
        let derivs = [dx, dy, inv_dx, inv_dy];

        let (clipped_line, top_is_on_left_edge, bot_is_on_left_edge) = Tile::clip_to_tile(
            line,
            tile_bounds,
            derivs,
            tile.intersection_mask(),
            canonical_x_dir,
            CANONICAL_Y_DIR,
        );
        let p_top = clipped_line.p0;
        let p_bot = clipped_line.p1;

        // Since the coordinates of the left edge touch are tile local, i.e. [0.0, 16.0], a
        // perfectly pixel aligned left edge intersection (e.g. y == 1.0) is exactly representable,
        // so `ceil` here and `floor` in the sidedness calculations detect pixel alignment.
        if tile.has_left_intersection() {
            let y_edge = if p_top.x < p_bot.x { p_top.y } else { p_bot.y };
            self.fill_left(y_edge, canonical_x_dir);
        }

        // Ignore perfectly horizontal lines that lie exactly on pixel boundaries, as their winding
        // contribution is accounted for by `fill_left`.
        if dy.abs() < STRIP_EPSILON && p_top.y == p_top.y.floor() {
            return;
        }

        // Use the tile intersection points to get the vertical span in (integer) pixels.
        let start_y = p_top.y.floor() as i32;
        let end_y = p_bot.y.ceil() as i32;
        if start_y < end_y {
            let is_only_row = start_y == end_y - 1;
            let row_int = Self::find_row_intersections(p_top, p_bot, dxdy, start_y, end_y);
            let step_params = self.compute_line_step_params(p_top, p_bot, dx, dy);
            let sorted_x_dir = step_params.sorted_x_dir;

            // First, and possibly only row.
            {
                let p_top_y = p_top.y;
                let p_bot_y = if is_only_row {
                    p_bot.y
                } else {
                    (start_y + 1) as f32
                };

                let crossed_top = p_top_y == p_top_y.floor();
                let default_invert = !crossed_top && sorted_x_dir;

                // The starting pixel either inverts (when it starts on the left edge) or
                // truncates the samples above its start, never both.
                let mut start_mask_val = FULL_MASK;
                let start_invert = top_is_on_left_edge && !crossed_top;
                if !top_is_on_left_edge {
                    start_mask_val = Self::get_truncation_mask::<true>(p_top_y, start_y as f32);
                }

                let mut end_mask_val = FULL_MASK;
                if is_only_row && !bot_is_on_left_edge {
                    end_mask_val = Self::get_truncation_mask::<false>(p_bot_y, start_y as f32);
                }

                let left_invert = if sorted_x_dir {
                    start_invert
                } else {
                    default_invert
                };
                let right_invert = if sorted_x_dir {
                    default_invert
                } else {
                    start_invert
                };

                let left_mask = if sorted_x_dir {
                    start_mask_val
                } else {
                    end_mask_val
                };
                let right_mask = if sorted_x_dir {
                    end_mask_val
                } else {
                    start_mask_val
                };
                self.process_row_span::<CANONICAL_Y_DIR>(
                    start_y,
                    left_mask,
                    right_mask,
                    left_invert,
                    default_invert,
                    right_invert,
                    crossed_top,
                    &row_int,
                    &step_params,
                );
            }

            // Middle rows.
            for row in start_y + 1..end_y - 1 {
                self.process_row_span::<CANONICAL_Y_DIR>(
                    row,
                    FULL_MASK,
                    FULL_MASK,
                    false,
                    false,
                    false,
                    true,
                    &row_int,
                    &step_params,
                );
            }

            // Bottom row, if it exists.
            if !is_only_row {
                let last_y = end_y - 1;
                let mut end_mask_last = FULL_MASK;
                if !bot_is_on_left_edge {
                    end_mask_last = Self::get_truncation_mask::<false>(p_bot.y, last_y as f32);
                }

                let left_mask = if sorted_x_dir {
                    FULL_MASK
                } else {
                    end_mask_last
                };
                let right_mask = if sorted_x_dir {
                    end_mask_last
                } else {
                    FULL_MASK
                };
                self.process_row_span::<CANONICAL_Y_DIR>(
                    last_y,
                    left_mask,
                    right_mask,
                    false,
                    false,
                    false,
                    true,
                    &row_int,
                    &step_params,
                );
            }
        }
    }

    /// Add (nonzero) or XOR (even-odd) `fill` into `target`. Skia's `ApplyWinding`.
    #[inline(always)]
    fn apply_winding(target: &mut SwarPixel, fill: SwarPixel) {
        if IS_WINDING {
            *target = swar::add(*target, fill);
        } else {
            *target ^= fill;
        }
    }

    /// Evaluate the subsamples against the winding rule to determine which are "active"
    /// (covered), as a sample mask. Skia's `GetActivesWide`.
    ///
    /// For even-odd no processing is required. For nonzero, we compare the subsample winding
    /// against the empty SWAR value.
    #[inline(always)]
    fn get_actives_wide(v: SwarPixel) -> SubSampleType {
        if IS_WINDING {
            swar::ne_mask(v, swar::splat(Self::INITIAL_WINDING))
        } else {
            debug_assert_eq!(v & !swar::ONES, 0, "even-odd lanes must be 0 or 1");
            swar::odd_mask(v)
        }
    }

    /// The x coordinate of the line at each pixel row boundary `k` (tile local), with the exact
    /// clipped endpoints at `start_y` and `end_y`.
    #[inline(always)]
    fn find_row_intersections(
        p_top: Point,
        p_bot: Point,
        dxdy: f32,
        start_y: i32,
        end_y: i32,
    ) -> [f32; TILE_HEIGHT + 1] {
        let mut row_int = [0.0; TILE_HEIGHT + 1];
        for (k, x) in row_int.iter_mut().enumerate() {
            *x = p_top.x + (k as f32 - p_top.y) * dxdy;
        }
        row_int[start_y as usize] = p_top.x;
        row_int[end_y as usize] = p_bot.x;
        row_int
    }

    /// When truncating, we mask away subsamples *above* the topmost point or subsamples *below*
    /// the bottommost point. (These can be the raw line endpoint or a tile edge intersection.)
    /// To do this, we rely on the properties of our LUT mask construction:
    ///
    /// 1. N-rooks subsample pattern: no two samples share the same y coordinate. Each of the 16
    ///    subsamples occupies its own distinct 1/16th vertical slice of the pixel, so the
    ///    fractional vertical distance inside the pixel maps linearly to the number of subsamples:
    ///    `16 * (p - row)` directly yields the bit shift amount.
    /// 2. Vertically sorted subsamples: the subsample mask bits are ordered top-to-bottom, with
    ///    the LSB corresponding to the topmost subsample in the pixel. So shifting left/right
    ///    corresponds to truncating top/bottom.
    ///
    /// Because traversal is top-to-bottom, `row` is the y coordinate of the top edge of the
    /// current pixel row, and `p - row` the fractional distance from the top edge down to `p`.
    /// The samples are at the centers of their slices, so a fractional distance > 0.5 slices must
    /// be crossing (and thus truncated) and vice versa, which is what rounding computes. Lines
    /// ending precisely at a slice midpoint round up and truncate, which is visually
    /// inconsequential.
    ///
    /// Left-shifting a full mask by the shift amount masks out the samples *above* `p` (start
    /// mask). For the end mask, the shifted mask is inverted, keeping the samples above `p`.
    #[inline(always)]
    fn get_truncation_mask<const IS_START: bool>(p: f32, row: f32) -> SubSampleType {
        let shift = ((NUM_SUB_SAMPLES as f32 * (p - row)).round() as u32).min(16);
        let shifted = u32::from(FULL_MASK) << shift;
        if IS_START {
            shifted as SubSampleType
        } else {
            !shifted as SubSampleType
        }
    }

    /// Computes the normalized parameter space (s, t) used to index into the precomputed MSAA
    /// look-up table:
    ///
    /// 0. DDA: because the line is straight, the change in `t` as we move one pixel right or down
    ///    is constant: `t(X, Y) = tBase + X * stepX + Y * stepY`.
    /// 1. Normalizing the slope: the LUT normalizes slopes into (0, 1] using `s = 1 / (m + 1)`
    ///    with `m = |dy/dx|`. Multiplying the numerator and denominator by `|dx|` handles vertical
    ///    lines: `s = |dx| / (|dy| + |dx|)`. The denominator is the Manhattan length `D` of the
    ///    line's rightward-pointing normal, `D = normalX + |normalY|`, so `s = |normalY| / D`.
    /// 2. DDA stepping: expanding `(x - (1 - t))(1 - s) - (y - t)s >= 0`, moving one pixel right
    ///    changes the value by `1 - s = normalX / D`, and moving one pixel down by
    ///    `-s = normalY / D` (if `dy > 0`, `normalY` is negative).
    /// 3. `tBase`: the normalized translation at the top left of the tile. The unnormalized line
    ///    offset at the origin is `-C`. For positive slopes, it is shifted by `normalX` to align
    ///    the phase with the LUT. For negative slopes, the LUT flips the y axis of the samples,
    ///    which shifts the offset by the full Manhattan distance instead.
    ///
    /// The steps are converted to 16.16 fixed point, pre-scaled by the LUT width, so that the LUT
    /// column of a pixel is `tFixed >> 16`.
    #[inline(always)]
    fn compute_line_step_params(
        &self,
        p_top: Point,
        p_bot: Point,
        dx: f32,
        dy: f32,
    ) -> LineStepParams<'a> {
        let mut normal_x = dy;
        let mut normal_y = -dx;

        // Force the normal vector to always point right (positive X). This ensures the returned
        // LUT mask emulates left-to-right scanline behavior.
        if normal_x < 0.0 {
            normal_x = -normal_x;
            normal_y = -normal_y;
        }

        // Compute the Manhattan distance of the normal vector.
        let d = normal_x + normal_y.abs();
        let inv_d = if d < STRIP_EPSILON { 0.0 } else { 1.0 / d };

        // In screen space (y goes down), a line that goes right-and-down (\) yields a negative
        // normal_y, which maps to a positive slope.
        let has_positive_slope = normal_y <= 0.0;

        // Compute the constant in the implicit line equation: Ax + By = C.
        let c = normal_x * p_top.x + normal_y * p_top.y;

        // Find `s`, and get the row associated with it in the LUT.
        let s = normal_y.abs() * inv_d;
        let half_height = (LUT_MASK_HEIGHT / 2) as i32;
        let lut_row_offset = ((s * half_height as f32).floor() as i32).clamp(0, half_height - 1);

        // If the slope is positive, shift the index into the bottom half of the LUT.
        let lut_row = if has_positive_slope {
            lut_row_offset + half_height
        } else {
            lut_row_offset
        };

        // Unlike the scalar version, we simply return the row of the LUT.
        let mask_lut: &'a LutArray = self.mask_lut;
        let mask_row_lut = &mask_lut[lut_row as usize];

        // DDA steps.
        let step_x = normal_x * inv_d;
        let step_y = normal_y * inv_d;

        // Calculate `tBase` at the top left of the tile. Substituting y = 1 - y' into the line
        // equation changes the translation offset from -C to (D - C). Thus, `tBase` uses
        // `normal_x` for positive slopes and `D` for negative slopes.
        let t_base = ((if has_positive_slope { normal_x } else { d }) - c) * inv_d;

        // We use 16.16 fixed-point arithmetic to avoid floating-point math and conversions inside
        // the inner loops: the upper 16 bits are the integer part and the lower 16 bits the
        // fraction. In addition to the 16.16 scale, we scale by the LUT's width (64) so that the
        // LUT column `u = floor(t * 64)` is `clamp(tFixed >> 16, 0, 63)`.
        //
        // The step and base values are bounded (the clipped line is within the tile), so plain
        // conversions without saturation checks suffice.
        const FIXED_MULT: f32 = LUT_MASK_WIDTH as f32 * 65536.0;
        let step_x_fixed = (step_x * FIXED_MULT) as i32;
        let step_y_fixed = (step_y * FIXED_MULT) as i32;
        let t_base_fixed = (t_base * FIXED_MULT) as i32;

        // NOTE: This is NOT the canonical x direction, this is the direction from the top point to
        // the bottom point.
        let sorted_x_dir = p_top.x <= p_bot.x;

        LineStepParams {
            mask_row_lut,
            step_x_fixed,
            step_y_fixed,
            t_base_fixed,
            sorted_x_dir,
        }
    }

    /// Account for the winding contribution of a line crossing the left edge of the tile: all
    /// pixel rows below the crossing are toggled.
    #[inline(always)]
    fn fill_left(&mut self, y_edge: f32, canonical_x_dir: bool) {
        let fill_byte = if IS_WINDING {
            if canonical_x_dir { 0xff } else { 1 }
        } else {
            1
        };
        let fill = swar::splat(fill_byte);
        let start_y = (y_edge.ceil() as i32).clamp(0, TILE_HEIGHT as i32) as usize;
        for row in &mut self.subsample_winding[start_y..] {
            for pixel in row {
                Self::apply_winding(pixel, fill);
            }
        }
    }

    #[inline(always)]
    fn process_pixel<const CANONICAL_Y_DIR: bool, const IS_EDGE_PIXEL: bool>(
        pixel: &mut SwarPixel,
        truncation_mask: SubSampleType,
        p_invert: SwarPixel,
        t_fixed: i32,
        mask_row_lut: &[SubSampleType; LUT_MASK_WIDTH],
    ) {
        // Shift right by 16 to extract the integer LUT column index `u = floor(t * 64)`.
        let column = (t_fixed >> 16).clamp(0, LUT_MASK_WIDTH_EXCL) as usize;
        let mut mask_val = mask_row_lut[column];

        // Apply the truncation mask if we're one of the candidate pixels.
        if IS_EDGE_PIXEL {
            mask_val &= truncation_mask;
        }

        // Convert the 16 packed coverage bits into 16 lanes: 0xff (-1) for covered lanes, 0x00
        // for uncovered ones.
        let cmp = swar::expand_mask(mask_val);
        let mut p_subsample_winding = *pixel;

        if IS_WINDING {
            // `cmp - invert` is -1/0 for covered/uncovered lanes, or 0/+1 when inverted.
            let p_res = swar::sub(cmp, p_invert);
            if CANONICAL_Y_DIR {
                p_subsample_winding = swar::sub(p_subsample_winding, p_res);
            } else {
                p_subsample_winding = swar::add(p_subsample_winding, p_res);
            }
        } else {
            let p_res = (cmp & swar::ONES) ^ p_invert;
            p_subsample_winding ^= p_res;
        }

        *pixel = p_subsample_winding;
    }

    // The inversion masks could maybe be moved into const generics, but for now simply expose
    // them as function arguments and rely on the compiler to optimize them.
    fn process_row_span<const CANONICAL_Y_DIR: bool>(
        &mut self,
        row: i32,
        left_mask: SubSampleType,
        right_mask: SubSampleType,
        left_invert: bool,
        mid_invert: bool,
        right_invert: bool,
        crossed_top: bool,
        row_int: &[f32; TILE_HEIGHT + 1],
        params: &LineStepParams<'_>,
    ) {
        let p_top_x = row_int[row as usize];
        let p_bot_x = row_int[row as usize + 1];

        let x_min = p_top_x.min(p_bot_x);
        let x_max = p_top_x.max(p_bot_x);
        let x_start = (x_min.floor() as i32).clamp(0, TILE_WIDTH as i32 - 1);
        let x_end = (x_max.floor() as i32).clamp(0, TILE_WIDTH as i32 - 1);

        // Compute the initial translation parameter `t_fixed` in 16.16 fixed-point format
        // (pre-scaled by 64 * 65536) at the starting pixel (x_start, row) of the span using:
        // t_fixed = t_base_fixed + step_y_fixed * row + step_x_fixed * x_start
        let mut t_fixed =
            params.t_base_fixed + params.step_y_fixed * row + params.step_x_fixed * x_start;
        let row_subsample_windings = &mut self.subsample_winding[row as usize];
        let (x_start, x_end) = (x_start as usize, x_end as usize);

        let invert_byte = swar::splat(if IS_WINDING { 0xff } else { 1 });
        let invert = |b: bool| if b { invert_byte } else { 0 };
        if x_start == x_end {
            let combined_mask = left_mask & right_mask;
            Self::process_pixel::<CANONICAL_Y_DIR, true>(
                &mut row_subsample_windings[x_start],
                combined_mask,
                invert(left_invert),
                t_fixed,
                params.mask_row_lut,
            );
        } else {
            Self::process_pixel::<CANONICAL_Y_DIR, true>(
                &mut row_subsample_windings[x_start],
                left_mask,
                invert(left_invert),
                t_fixed,
                params.mask_row_lut,
            );
            t_fixed += params.step_x_fixed;

            for pixel in &mut row_subsample_windings[x_start + 1..x_end] {
                Self::process_pixel::<CANONICAL_Y_DIR, false>(
                    pixel,
                    0,
                    invert(mid_invert),
                    t_fixed,
                    params.mask_row_lut,
                );
                t_fixed += params.step_x_fixed;
            }

            Self::process_pixel::<CANONICAL_Y_DIR, true>(
                &mut row_subsample_windings[x_end],
                right_mask,
                invert(right_invert),
                t_fixed,
                params.mask_row_lut,
            );
        }

        if crossed_top {
            let fill_byte = if IS_WINDING {
                if CANONICAL_Y_DIR { 1 } else { 0xff }
            } else {
                1
            };
            let fill = swar::splat(fill_byte);
            for pixel in &mut row_subsample_windings[x_end + 1..] {
                Self::apply_winding(pixel, fill);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::lut::msaa_lut;
    use super::super::reference::rasterize_path_reference;
    use super::super::sink::StripSink;
    use super::super::tests::{XorShift, decode, polygon, star_polygon};
    use super::*;

    /// Rasterize closed `lines` that lie strictly inside tile `(tx, ty)` with the processor alone
    /// (as the tiler would emit them: one tile per line, no intersection bits), seeded with
    /// `coarse` winding.
    fn rasterize_in_tile<const IS_WINDING: bool>(
        lines: &[Line],
        tx: u16,
        ty: u16,
        coarse: i32,
    ) -> [u32; MASK_WORDS_PER_TILE] {
        let mut processor = StripProcessor::<'_, IS_WINDING>::new(lines, msaa_lut());
        processor.set_coarse_winding(coarse);
        processor.clear_with_coarse_winding();
        let (x, y) = (f32::from(tx) * 16.0, f32::from(ty) * 16.0);
        let bounds = [Point::new(x, y), Point::new(x + 16.0, y + 16.0)];
        for idx in 0..lines.len() {
            processor.rasterize_line_to_tile(Tile::new(tx, ty, idx as u32, 0), bounds);
        }
        let mut masks = [0; MASK_WORDS_PER_TILE];
        processor.resolve_masks(&mut masks);
        masks
    }

    /// Per-pixel masks of tile `(tx, ty)` from the reference rasterizer.
    fn reference_tile(lines: &[Line], even_odd: bool, tx: usize, ty: usize) -> Vec<u16> {
        let (w, h) = (64, 64);
        let mut sink = StripSink::new(4);
        sink.begin_path(0);
        rasterize_path_reference(lines, even_odd, w, h, &mut sink);
        let px = decode(&sink.records, &sink.masks, 0, w, h);
        let mut out = vec![0; 256];
        for y in 0..16 {
            for x in 0..16 {
                out[y * 16 + x] = px[(ty * 16 + y) * 64 + tx * 16 + x];
            }
        }
        out
    }

    fn unpack(masks: &[u32; MASK_WORDS_PER_TILE]) -> Vec<u16> {
        masks
            .iter()
            .flat_map(|w| [*w as u16, (*w >> 16) as u16])
            .collect()
    }

    /// Returns (sum of |popcount difference|, sum of differing samples, max per pixel).
    fn diff(a: &[u16], b: &[u16]) -> (u32, u32, u32) {
        let mut total = (0, 0, 0);
        for (&a, &b) in a.iter().zip(b) {
            let d = a.count_ones().abs_diff(b.count_ones());
            total.0 += d;
            total.1 += (a ^ b).count_ones();
            total.2 = total.2.max((a ^ b).count_ones());
        }
        total
    }

    fn perimeter(lines: &[Line]) -> f32 {
        lines
            .iter()
            .map(|l| (l.p1.x - l.p0.x).hypot(l.p1.y - l.p0.y))
            .sum()
    }

    #[test]
    fn interior_rect_is_exact() {
        // Edges on pixel boundaries and at half pixels.
        for (x0, y0, x1, y1) in [
            (18.0, 19.0, 29.0, 27.0),
            (17.5, 18.5, 30.5, 31.5),
            (16.25, 17.75, 20.5, 30.0),
        ] {
            let lines = polygon(&[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]);
            let reversed = polygon(&[(x0, y0), (x0, y1), (x1, y1), (x1, y0)]);
            for lines in [lines, reversed] {
                let expected = reference_tile(&lines, false, 1, 1);
                let nz = unpack(&rasterize_in_tile::<true>(&lines, 1, 1, 0));
                let eo = unpack(&rasterize_in_tile::<false>(&lines, 1, 1, 0));
                assert_eq!(nz, expected, "nonzero {x0} {y0} {x1} {y1}");
                assert_eq!(eo, expected, "even-odd {x0} {y0} {x1} {y1}");
            }
        }
    }

    #[test]
    fn interior_rect_with_backdrop() {
        // A hole cut into a tile that is inside another shape (coarse winding 1): the clockwise
        // (in y-down) rect has winding -1 inside, so nonzero leaves a hole, as does even-odd.
        let lines = polygon(&[(20.0, 20.0), (27.5, 20.0), (27.5, 28.5), (20.0, 28.5)]);
        let inside = reference_tile(&lines, false, 1, 1);
        let nz = unpack(&rasterize_in_tile::<true>(&lines, 1, 1, 1));
        let eo = unpack(&rasterize_in_tile::<false>(&lines, 1, 1, 1));
        let nz2 = unpack(&rasterize_in_tile::<true>(&lines, 1, 1, 2));
        let holes: Vec<u16> = inside.iter().map(|m| !m).collect();
        assert!(inside.contains(&0xffff), "{inside:?}");
        assert_eq!(nz, holes);
        assert_eq!(eo, holes);
        // With backdrop 2, the winding is 1 or 2 everywhere.
        assert!(nz2.iter().all(|&m| m == 0xffff), "{nz2:?}");
    }

    #[test]
    fn interior_polygons_match_reference() {
        let mut rng = XorShift(0x5eed_1234_abcd_0001);
        let mut stats = (0, 0, 0);
        let mut total_perimeter = 0.0;
        for i in 0..400 {
            let n = 3 + i % 9;
            let lines = if i % 2 == 0 {
                let r = 1.0 + 6.5 * rng.next_f32();
                star_polygon(&mut rng, 24.0, 24.0, r, n).0
            } else {
                // Random, generally self-intersecting polygon.
                let pts: Vec<(f32, f32)> = (0..n)
                    .map(|_| (16.5 + 15.0 * rng.next_f32(), 16.5 + 15.0 * rng.next_f32()))
                    .collect();
                polygon(&pts)
            };
            let perimeter = perimeter(&lines);
            for even_odd in [false, true] {
                let expected = reference_tile(&lines, even_odd, 1, 1);
                let got = if even_odd {
                    unpack(&rasterize_in_tile::<false>(&lines, 1, 1, 0))
                } else {
                    unpack(&rasterize_in_tile::<true>(&lines, 1, 1, 0))
                };
                let (d, x, m) = diff(&got, &expected);
                assert!(
                    d as f32 <= perimeter + 4.0,
                    "polygon {i} even_odd {even_odd}: |popcount diff| {d} (xor {x}, max {m}) \
                     perimeter {perimeter}: {lines:?}"
                );
                stats = (stats.0 + d, stats.1 + x, stats.2.max(m));
                total_perimeter += perimeter;
            }
        }
        println!(
            "interior polygons: |popcount diff| {:.4}/px of perimeter, xor {:.4}/px, max {}",
            stats.0 as f32 / total_perimeter,
            stats.1 as f32 / total_perimeter,
            stats.2
        );
    }
}
