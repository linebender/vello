// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Rendering strips.

use crate::flatten::Line;
use crate::peniko::Fill;
use crate::tile::{Tile, Tiles};
use crate::util::f32_to_u8;
use alloc::vec::Vec;
use fearless_simd::*;

#[derive(Clone, Copy)]
struct GenericStripKernel;

trait StripKernel<S: Simd>: Copy {
    fn update_coverage(
        s: S,
        location_winding: &mut [f32x4<S>; Tile::WIDTH as usize],
        p0_x: f32,
        p0_y: f32,
        p1_x: f32,
        p1_y: f32,
    ) -> (f32x4<S>, i32);

    fn alpha_to_u8(s: S, values: [f32x4<S>; Tile::WIDTH as usize]) -> u8x16<S>;
}

trait F32x4MaxExt {
    fn max_if_first_nan_take_second(self, rhs: Self) -> Self;
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl<S: Simd> F32x4MaxExt for f32x4<S> {
    #[inline(always)]
    fn max_if_first_nan_take_second(self, rhs: Self) -> Self {
        self.max(rhs)
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
impl<S: Simd> F32x4MaxExt for f32x4<S> {
    #[inline(always)]
    fn max_if_first_nan_take_second(self, rhs: Self) -> Self {
        self.max_precise(rhs)
    }
}

impl<S: Simd> StripKernel<S> for GenericStripKernel {
    #[inline(always)]
    fn update_coverage(
        s: S,
        location_winding: &mut [f32x4<S>; Tile::WIDTH as usize],
        p0_x: f32,
        p0_y: f32,
        p1_x: f32,
        p1_y: f32,
    ) -> (f32x4<S>, i32) {
        let (line_top_y, line_top_x, line_bottom_y, line_bottom_x, sign, sign_f32) = if p0_y < p1_y
        {
            (p0_y, p0_x, p1_y, p1_x, -1, -1.0)
        } else {
            (p1_y, p1_x, p0_y, p0_x, 1, 1.0)
        };

        let y_slope = (line_bottom_y - line_top_y) / (line_bottom_x - line_top_x);
        let x_slope = 1. / y_slope;

        let line_top_y = f32x4::splat(s, line_top_y);
        let line_bottom_y = f32x4::splat(s, line_bottom_y);
        let line_px_base_yx = line_top_y.mul_add(-x_slope, line_top_x);
        let px_top_y = f32x4::simd_from(s, [0., 1., 2., 3.]);
        let px_bottom_y = 1. + px_top_y;
        let ymin = line_top_y.max(px_top_y);
        let ymax = line_bottom_y.min(px_bottom_y);
        let mut acc = f32x4::splat(s, 0.0);

        for (x_idx, winding) in location_winding.iter_mut().enumerate() {
            let px_left_x = f32x4::splat(s, x_idx as f32);
            let px_right_x = 1.0 + px_left_x;
            let line_px_left_y = (px_left_x - line_top_x)
                .mul_add(y_slope, line_top_y)
                .max_if_first_nan_take_second(ymin)
                .min(ymax);
            let line_px_right_y = (px_right_x - line_top_x)
                .mul_add(y_slope, line_top_y)
                .max_if_first_nan_take_second(ymin)
                .min(ymax);
            let line_px_left_yx = line_px_left_y.mul_add(x_slope, line_px_base_yx);
            let line_px_right_yx = line_px_right_y.mul_add(x_slope, line_px_base_yx);
            let h = (line_px_right_y - line_px_left_y).abs();
            let area = h * (line_px_right_yx + line_px_left_yx).mul_add(-0.5, px_right_x);
            *winding += area.mul_add(sign_f32, acc);
            acc = h.mul_add(sign_f32, acc);
        }

        (acc, sign)
    }

    #[inline(always)]
    fn alpha_to_u8(s: S, values: [f32x4<S>; Tile::WIDTH as usize]) -> u8x16<S> {
        let p1 = s.combine_f32x4(values[0], values[1]);
        let p2 = s.combine_f32x4(values[2], values[3]);
        f32_to_u8(s.combine_f32x8(p1, p2))
    }
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct Avx2AsmStripKernel;

#[cfg(target_arch = "x86_64")]
impl StripKernel<Avx2> for Avx2AsmStripKernel {
    #[inline(always)]
    fn update_coverage(
        _s: Avx2,
        location_winding: &mut [f32x4<Avx2>; Tile::WIDTH as usize],
        p0_x: f32,
        p0_y: f32,
        p1_x: f32,
        p1_y: f32,
    ) -> (f32x4<Avx2>, i32) {
        let (line_top_y, line_top_x, line_bottom_y, line_bottom_x, sign, sign_f32) = if p0_y < p1_y
        {
            (p0_y, p0_x, p1_y, p1_x, -1, -1.0)
        } else {
            (p1_y, p1_x, p0_y, p0_x, 1, 1.0)
        };
        let delta_y = line_bottom_y - line_top_y;
        let delta_x = line_bottom_x - line_top_x;
        // These divisions are independent, unlike `x_slope = 1.0 / y_slope`.
        let y_slope = delta_y / delta_x;
        let x_slope = delta_x / delta_y;
        let acc = crate::strip_avx2::update_coverage(
            location_winding,
            line_top_y,
            line_top_x,
            line_bottom_y,
            y_slope,
            x_slope,
            sign_f32,
        );
        (acc, sign)
    }

    #[inline(always)]
    fn alpha_to_u8(_s: Avx2, values: [f32x4<Avx2>; Tile::WIDTH as usize]) -> u8x16<Avx2> {
        crate::strip_avx2::alpha_to_u8(values)
    }
}

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

    /// Creates a sentinel strip.
    pub fn sentinel(y: u16, alpha_idx: u32) -> Self {
        Self::new(u16::MAX, y, alpha_idx, false)
    }

    /// Return whether the strip is a sentinel strip.
    pub fn is_sentinel(&self) -> bool {
        self.x == u16::MAX
    }

    /// Return the y coordinate of the strip, in strip units.
    pub fn strip_y(&self) -> u16 {
        self.y / Tile::HEIGHT
    }

    /// Returns the horizontal pixel width of this strip.
    ///
    /// **IMPORTANT**: This assumes that the `next` is actually the next adjacent strip
    /// to `self`, otherwise this method will return a garbage value!
    pub fn width_to(&self, next: &Self) -> u16 {
        let col = self.alpha_idx() / u32::from(Tile::HEIGHT);
        let next_col = next.alpha_idx() / u32::from(Tile::HEIGHT);
        next_col.saturating_sub(col) as u16
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

    /// When early culling is active, geometry fully to the left of the viewport creates no tiles.
    /// However, if that geometry has a non-zero winding (e.g. a large shape surrounding the
    /// viewport), then we must output strips for those fills.
    ///
    /// We reconstruct this "background" fill using `row_windings` (the winding at x=0) to emit solid
    /// strips for:
    ///      1. All rows vertically above the first visible tile.
    ///      2. 'Captive' rows between two tile-containing rows.
    ///      3. All rows vertically below the last visible tile.
    #[inline(always)]
    fn emit_culled_background<F>(
        start: u16,
        end: u16,
        viewport_width: u16,
        strips: &mut Vec<Self>,
        alphas: &mut Vec<u8>,
        windings: &crate::tile::CulledWindings,
        mut should_fill: F,
    ) where
        F: FnMut(i32) -> bool,
    {
        windings.for_active_rows_in_range(start as usize, end as usize, |row| {
            if should_fill(windings.coarse[row] as i32) {
                let y_pos = row as u16 * Tile::HEIGHT;
                strips.push(Self::new(0, y_pos, alphas.len() as u32, false));
                // TODO: Would be nice to get rid of this, but the current clipping code only
                // allows zero-width strips as a row terminator, not in-between.
                alphas.extend([255_u8; Tile::HEIGHT as usize * Tile::WIDTH as usize]);
                strips.push(Self::new(viewport_width, y_pos, alphas.len() as u32, true));
            }
        });
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
    #[cfg(target_arch = "x86_64")]
    if let Some(avx2) = level.as_avx2() {
        avx2.vectorize(
            #[inline(always)]
            || {
                render_impl(
                    avx2,
                    tiles,
                    strip_buf,
                    alpha_buf,
                    fill_rule,
                    aliasing_threshold,
                    lines,
                    Avx2AsmStripKernel,
                );
            },
        );
        return;
    }

    dispatch!(level, simd => render_impl(simd,
                                         tiles,
                                         strip_buf,
                                         alpha_buf,
                                         fill_rule,
                                         aliasing_threshold,
                                         lines,
                                         GenericStripKernel));
}

#[inline(always)]
fn render_impl<S: Simd, K: StripKernel<S>>(
    s: S,
    tiles: &Tiles,
    strip_buf: &mut Vec<Strip>,
    alpha_buf: &mut Vec<u8>,
    fill_rule: Fill,
    aliasing_threshold: Option<u8>,
    lines: &[Line],
    _kernel: K,
) {
    let row_windings = &tiles.windings.coarse;
    let has_culled_tiles = tiles.has_culled_tiles();
    let viewport_width = tiles
        .width()
        // We need to make sure strips are tile-aligned.
        .checked_next_multiple_of(Tile::WIDTH)
        .unwrap_or(u16::MAX);
    let strip_start = strip_buf.len();
    let maybe_emit_sentinel_strip = |strip_buf: &mut Vec<Strip>, alpha_buf: &Vec<u8>| {
        // Emit the final sentinel strip, if we produced at least one strip.
        if let Some(last_y) = strip_buf[strip_start..].last().map(|s| s.y) {
            strip_buf.push(Strip::sentinel(last_y, alpha_buf.len() as u32));
        }
    };

    // If no tiles were culled and the tile buffer is empty, we can simply exit. If tiles were
    // culled, the tile buffer may be empty but there may be winding produced by culled geometry
    // left of the viewport that must be checked for filling.
    if !has_culled_tiles && tiles.is_empty() {
        return;
    }

    let should_fill = |winding: i32| match fill_rule {
        Fill::NonZero => winding != 0,
        Fill::EvenOdd => winding % 2 != 0,
    };

    // Helper to handle "captive strips". When a row has tiles, but the first tile
    // is not at the left edge of the viewport (x != 0), we must emit a solid strip
    // from x=0 to that tile if the coarse winding dictates a fill.
    let emit_captive_strip =
        |y: u16, is_left_viewport: bool, strips: &mut Vec<Strip>, alphas: &mut Vec<u8>| {
            let coarse_wd = tiles.windings.coarse[y as usize] as i32;

            if should_fill(coarse_wd) && !is_left_viewport {
                strips.push(Strip::new(0, y * Tile::HEIGHT, alphas.len() as u32, false));
                alphas.extend([255_u8; Tile::HEIGHT as usize * Tile::WIDTH as usize]);
            }

            let mut acc = f32x4::splat(s, coarse_wd as f32);
            if is_left_viewport {
                let fine_winding: f32x4<_> = tiles.windings.partial[y as usize].simd_into(s);
                acc += fine_winding;
            }

            (coarse_wd, acc)
        };

    // The accumulated tile winding delta. A line that crosses the top edge of a tile
    // increments the delta if the line is directed upwards, and decrements it if goes
    // downwards. Horizontal lines leave it unchanged.
    let mut winding_delta: i32 = 0;

    // The previous tile visited.
    let mut prev_tile = if has_culled_tiles && tiles.is_empty() {
        Tile::SENTINEL
    } else {
        *tiles.get(0)
    };

    // The accumulated (fractional) winding of the tile-sized location we're currently at.
    // Note multiple tiles can be at the same location.
    // Note that we are also implicitly assuming here that the tile height exactly fits into a
    // SIMD vector (i.e. 128 bits).
    let mut location_winding = [f32x4::splat(s, 0.0); Tile::WIDTH as usize];
    // The accumulated (fractional) windings at this location's right edge. When we move to the
    // next location, this is splatted to that location's starting winding.
    let mut accumulated_winding = f32x4::splat(s, 0.0);

    let left_viewport = prev_tile.x == 0;
    if has_culled_tiles {
        let row_max = prev_tile.y.min(row_windings.len() as u16);
        Strip::emit_culled_background(
            0,
            row_max,
            viewport_width,
            strip_buf,
            alpha_buf,
            &tiles.windings,
            should_fill,
        );
        if tiles.is_empty() {
            maybe_emit_sentinel_strip(strip_buf, alpha_buf);

            return;
        }
        let (wd, acc) = emit_captive_strip(prev_tile.y, left_viewport, strip_buf, alpha_buf);
        winding_delta = wd;
        accumulated_winding = acc;
        location_winding = [accumulated_winding; Tile::WIDTH as usize];
    }

    // The strip we're building.
    let mut strip = Strip::new(
        prev_tile.x * Tile::WIDTH,
        prev_tile.y * Tile::HEIGHT,
        alpha_buf.len() as u32,
        should_fill(winding_delta) && !left_viewport,
    );

    for (tile_idx, tile) in tiles.iter().copied().chain([Tile::SENTINEL]).enumerate() {
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

            let mut u8_vals = K::alpha_to_u8(s, location_winding);

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
            let left_viewport = tile.x == 0;
            if !prev_tile.same_row(&tile) {
                // Emit a final strip in the row if there is non-zero winding for the sparse fill
                if winding_delta != 0 {
                    strip_buf.push(Strip::new(
                        viewport_width,
                        prev_tile.y * Tile::HEIGHT,
                        alpha_buf.len() as u32,
                        should_fill(winding_delta),
                    ));
                }

                // Logic identical to the start (see above): fill any vertical gaps (empty rows)
                // between the previous and current tile using the row windings.
                if has_culled_tiles && !is_sentinel {
                    Strip::emit_culled_background(
                        prev_tile.y + 1,
                        tile.y,
                        viewport_width,
                        strip_buf,
                        alpha_buf,
                        &tiles.windings,
                        should_fill,
                    );

                    let (wd, acc) = emit_captive_strip(tile.y, left_viewport, strip_buf, alpha_buf);
                    winding_delta = wd;
                    accumulated_winding = acc;
                } else {
                    winding_delta = 0;
                    accumulated_winding = f32x4::splat(s, 0.0);
                };

                #[expect(clippy::needless_range_loop, reason = "dimension clarity")]
                for x in 0..Tile::WIDTH as usize {
                    location_winding[x] = accumulated_winding;
                }
            } else {
                // Note: this fill is mathematically not necessary. It provides a way to reduce
                // accumulation of float rounding errors.
                accumulated_winding = f32x4::splat(s, winding_delta as f32);
            }

            if is_sentinel {
                break;
            }

            strip = Strip::new(
                tile.x * Tile::WIDTH,
                tile.y * Tile::HEIGHT,
                alpha_buf.len() as u32,
                should_fill(winding_delta) && !left_viewport,
            );
        }
        prev_tile = tile;

        // TODO: horizontal geometry has no impact on winding. This branch will be removed when
        // horizontal geometry is culled at the tile-generation stage.
        if p0_y == p1_y {
            continue;
        }

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

        let (acc, sign) = K::update_coverage(s, &mut location_winding, p0_x, p0_y, p1_x, p1_y);
        winding_delta += sign * i32::from(tile.winding());
        accumulated_winding += acc;
    }

    if has_culled_tiles {
        Strip::emit_culled_background(
            (prev_tile.y + 1).min(row_windings.len() as u16),
            row_windings.len() as u16,
            viewport_width,
            strip_buf,
            alpha_buf,
            &tiles.windings,
            should_fill,
        );
    }

    maybe_emit_sentinel_strip(strip_buf, alpha_buf);
}
