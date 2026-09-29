// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Fast pixel-aligned rectangle rendering directly into strips.

use crate::kurbo::Rect;
#[cfg(not(feature = "std"))]
use crate::kurbo::common::FloatFuncs as _;
use crate::simd::element_wise_splat;
use crate::strip::Strip;
use crate::tile::Tile;
use crate::util::f32_to_u8;
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

/// Generates strip data for an axis-aligned rectangle.
///
/// # Strip layout strategy
///
/// Tile rows are classified into two kinds:
///
/// - **Edge rows** (top/bottom of rect): the rect boundary crosses partway
///   through the tile vertically, so individual pixels need per-cell alpha.
///   We emit a *single wide strip* spanning all tile columns, with alpha =
///   `x_alpha` * `y_alpha` (so the intersection of the alpha mask in each direction).
///
/// - **Interior rows**: every pixel in the tile has full vertical coverage,
///   so we only need to handle the left and right partial-column edges.
///   We emit a **left edge strip** (with its x-alpha mask) and, when the rect
///   spans more than one tile column, a **right edge strip** with `fill_gap =
///   true` so the renderer fills solid 0xFF between them.
///
/// The x-alpha masks for the left/right edge tiles are y-independent, so they
/// are precomputed once and reused across all interior rows.
#[inline(always)]
fn render_impl<S: Simd>(s: S, rect: Rect, strip_buf: &mut Vec<Strip>, alpha_buf: &mut Vec<u8>) {
    if rect.is_zero_area() {
        return;
    }

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

    let y0 = u32::from((px_y0 / Tile::HEIGHT) * Tile::HEIGHT);
    let y1 = (u32::from(px_y1) + u32::from(Tile::HEIGHT - 1)) / u32::from(Tile::HEIGHT)
        * u32::from(Tile::HEIGHT);
    // Exclusive end of the right-edge tile, widened to avoid overflow.
    let x_end = u32::from(right_tile_x) + u32::from(Tile::WIDTH);

    if x_end <= u32::from(left_tile_x) || y1 <= y0 {
        return;
    }

    let tile_start_y = y0 / u32::from(Tile::HEIGHT);
    let tile_end_y = y1 / u32::from(Tile::HEIGHT);

    // A right strip is only needed when the rect spans more than one tile column.
    let needs_right_strip = right_tile_x > left_tile_x;

    let left_x_cov = coverage(s, left_tile_x, rect_x0, rect_x1);
    let right_x_cov = coverage(s, right_tile_x, rect_x0, rect_x1);
    // Edge rows span every tile column; interior rows need only the two x masks.
    let render_edge_row = {
        #[inline(always)]
        |strip_y: u16, strip_buf: &mut Vec<Strip>, alpha_buf: &mut Vec<u8>| {
            let alpha_start = alpha_buf.len() as u32;
            let y_cov = coverage(s, strip_y, rect_y0, rect_y1);
            let left_alpha = combined_tile_alpha(s, &left_x_cov, &y_cov);
            alpha_buf.extend_from_slice(left_alpha.as_slice());

            if needs_right_strip {
                let interior_tile_count =
                    usize::from((right_tile_x - left_tile_x) / Tile::WIDTH - 1);
                let interior_alpha: [u8; 16] =
                    combined_tile_alpha(s, &[1.0; Tile::WIDTH as usize], &y_cov).into();
                alpha_buf
                    .extend(core::iter::repeat_n(interior_alpha, interior_tile_count).flatten());

                let right_alpha = combined_tile_alpha(s, &right_x_cov, &y_cov);
                alpha_buf.extend_from_slice(right_alpha.as_slice());
            }

            strip_buf.push(Strip::new(left_tile_x, strip_y, alpha_start, false));
        }
    };

    let mut interior_start_y = tile_start_y;
    let mut interior_end_y = tile_end_y;
    if (y0 as f32) < rect_y0 {
        render_edge_row(y0 as u16, strip_buf, alpha_buf);
        interior_start_y += 1;
    }
    // A single tile row may contain both edges. Do not emit that row twice.
    if interior_start_y < interior_end_y && rect_y1 < y1 as f32 {
        interior_end_y -= 1;
    }

    let interior_row_count = (interior_end_y - interior_start_y) as usize;
    if interior_row_count > 0 {
        let alpha_start = alpha_buf.len() as u32;
        let tile_alpha_len = u32::from(Tile::WIDTH) * u32::from(Tile::HEIGHT);
        let left_x_mask = alpha_mask_from_x_coverage(s, &left_x_cov);

        if needs_right_strip {
            let right_x_mask = alpha_mask_from_x_coverage(s, &right_x_cov);
            let row_alpha: [u8; 32] = s.combine_u8x16(left_x_mask, right_x_mask).into();
            alpha_buf.extend(core::iter::repeat_n(row_alpha, interior_row_count).flatten());
            strip_buf.extend((interior_start_y..interior_end_y).flat_map(|tile_y| {
                let strip_y = (tile_y * u32::from(Tile::HEIGHT)) as u16;
                let alpha_idx = alpha_start + (tile_y - interior_start_y) * 2 * tile_alpha_len;
                [
                    Strip::new(left_tile_x, strip_y, alpha_idx, false),
                    // Fill the fully covered gap between the two edge tiles.
                    Strip::new(right_tile_x, strip_y, alpha_idx + tile_alpha_len, true),
                ]
            }));
        } else {
            let left_x_mask: [u8; 16] = left_x_mask.into();
            alpha_buf.extend(core::iter::repeat_n(left_x_mask, interior_row_count).flatten());
            strip_buf.extend((interior_start_y..interior_end_y).map(|tile_y| {
                let strip_y = (tile_y * u32::from(Tile::HEIGHT)) as u16;
                let alpha_idx = alpha_start + (tile_y - interior_start_y) * tile_alpha_len;
                Strip::new(left_tile_x, strip_y, alpha_idx, false)
            }));
        }
    }

    if interior_end_y < tile_end_y {
        render_edge_row((y1 - u32::from(Tile::HEIGHT)) as u16, strip_buf, alpha_buf);
    }

    // Sentinel strip: marks the end of the strip list for this shape.
    let last_strip_y = ((tile_end_y - 1) * u32::from(Tile::HEIGHT)) as u16;
    strip_buf.push(Strip::sentinel(last_strip_y, alpha_buf.len() as u32));
}

/// Compute fractional pixel coverage for four consecutive pixels starting at `start`.
#[inline(always)]
fn coverage<S: Simd>(s: S, start: u16, rect_lo: f32, rect_hi: f32) -> [f32; 4] {
    let px = f32x4::splat(s, f32::from(start)) + f32x4::from_slice(s, &[0.0, 1.0, 2.0, 3.0]);
    let lo = f32x4::splat(s, rect_lo).max(px);
    let hi = f32x4::splat(s, rect_hi).min(px + 1.0);
    (hi - lo).max(0.0).min(1.0).into()
}

/// Build an alpha mask for the 4x4 tile from the given horizontal coverages,
/// splatting them across the other dimension.
#[inline(always)]
fn alpha_mask_from_x_coverage<S: Simd>(s: S, cov: &[f32; Tile::WIDTH as usize]) -> u8x16<S> {
    let alpha = f32x4::from_slice(s, cov)
        .mul_add(255.0, 0.5)
        .to_int::<u32x4<S>>();
    // Each coverage is in [0, 1], so its alpha fits in a byte. Replicate that
    // byte across its u32 lane to fill the four rows of the column.
    (alpha * 0x0101_0101_u32).to_bytes()
}

/// Compute the alphas for a single 4x4 tile, taking horizontal as well as vertical coverage
/// of the rectangle into account.
#[inline(always)]
fn combined_tile_alpha<S: Simd>(
    s: S,
    x_cov: &[f32; Tile::WIDTH as usize],
    y_cov: &[f32; Tile::HEIGHT as usize],
) -> u8x16<S> {
    // Tiles are stored in column-major order, so each x coverage is repeated
    // for all rows, and the y coverages are repeated for all columns.
    let x_cov = element_wise_splat(s, f32x4::from_slice(s, x_cov));
    let y_cov = f32x16::block_splat(f32x4::from_slice(s, y_cov));

    f32_to_u8((x_cov * y_cov).mul_add(255.0, 0.5))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn render_edge_row_at_u16_right_edge() {
        let mut strips = Vec::new();
        let mut alphas = Vec::new();
        let rect = Rect::new(f64::from(u16::MAX - 3), 0.5, f64::from(u16::MAX), 3.5);

        render(Level::baseline(), rect, &mut strips, &mut alphas);

        assert_eq!(strips.len(), 2);
        assert_eq!(strips[0].x, u16::MAX - 3);
        assert_eq!(strips[0].alpha_idx(), 0);
        assert_eq!(
            alphas.len(),
            usize::from(Tile::WIDTH) * usize::from(Tile::HEIGHT)
        );
        assert!(strips[1].is_sentinel());
    }

    #[test]
    fn render_edge_row_at_u16_bottom_edge() {
        let mut strips = Vec::new();
        let mut alphas = Vec::new();
        let rect = Rect::new(0.5, f64::from(u16::MAX - 3), 3.5, f64::from(u16::MAX));

        render(Level::baseline(), rect, &mut strips, &mut alphas);

        assert_eq!(strips.len(), 2);
        assert_eq!(strips[0].y, u16::MAX - 3);
        assert_eq!(strips[0].alpha_idx(), 0);
        assert_eq!(
            alphas.len(),
            usize::from(Tile::WIDTH) * usize::from(Tile::HEIGHT)
        );
        assert!(strips[1].is_sentinel());
    }
}
