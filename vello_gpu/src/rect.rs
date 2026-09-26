// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Helpers for decomposing rectangles used by the rectangle fast path.

use vello_common::geometry::RectU16;
use vello_common::kurbo::Rect;
use vello_common::rect::{coverage_to_alpha, pixel_coverage};

/// The threshold of the rectangle size after which a rectangle should be split up
/// into multiple smaller ones.
const LARGE_RECT_SPLIT_THRESHOLD: u16 = 32;

/// Integer rectangle geometry and the alphas of its boundary pixels (see `rect.wesl`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RectPart {
    /// Pixel-aligned bounds of this rectangle part.
    pub(crate) rect: RectU16,
    /// Alphas of the first column, last column, first row and last row, from the low byte up.
    pub(crate) alphas: u32,
    /// Corrections of the top-left, top-right, bottom-left and bottom-right pixel, from the low
    /// bits up, as 2-bit two's complement.
    ///
    /// Each is the exact alpha minus [`combine_alphas`] of the column and row alphas, which is
    /// always -1, 0 or 1.
    pub(crate) corner_corrections: u8,
}

impl RectPart {
    /// A part whose pixels are all fully covered.
    pub(crate) fn full(rect: RectU16) -> Self {
        Self {
            rect,
            alphas: u32::MAX,
            corner_corrections: 0,
        }
    }

    /// Compute the boundary alphas of `rect` for the rectangle `[x0, x1] x [y0, y1]`.
    fn new(rect: RectU16, x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        let cov_x = [rect.x0, rect.x1 - 1].map(|px| pixel_coverage(f32::from(px), x0, x1));
        let cov_y = [rect.y0, rect.y1 - 1].map(|py| pixel_coverage(f32::from(py), y0, y1));
        let alpha_x = cov_x.map(coverage_to_alpha);
        let alpha_y = cov_y.map(coverage_to_alpha);

        let mut corner_corrections = 0;
        for (row, (cy, ay)) in cov_y.into_iter().zip(alpha_y).enumerate() {
            for (col, (cx, ax)) in cov_x.into_iter().zip(alpha_x).enumerate() {
                let exact = coverage_to_alpha(cx * cy);
                let combined = combine_alphas(ax, ay);
                debug_assert!(
                    exact.abs_diff(combined) <= 1,
                    "corner alpha {exact} is more than one step away from {combined}"
                );
                corner_corrections |=
                    (exact.wrapping_sub(combined) & 0b11) << (2 * (col + 2 * row));
            }
        }

        Self {
            rect,
            alphas: u32::from_le_bytes([alpha_x[0], alpha_x[1], alpha_y[0], alpha_y[1]]),
            corner_corrections,
        }
    }

    /// Whether all pixels of this part are fully covered.
    ///
    /// Checking the boundary alphas is not enough: edges less than half an alpha step away from
    /// the pixel grid have alphas of 255, but their corner can still round to 254.
    pub(crate) fn is_full(&self) -> bool {
        self.alphas == u32::MAX && self.corner_corrections == 0
    }

    pub(crate) fn shift(self, shift: (i32, i32)) -> Self {
        Self {
            rect: self.rect.shift(shift),
            ..self
        }
    }
}

/// `round(x * y / 255)`, computed like the shader does.
#[expect(clippy::cast_possible_truncation, reason = "the result is at most 255")]
fn combine_alphas(x: u8, y: u8) -> u8 {
    let p = u32::from(x) * u32::from(y) + 128;
    ((p + (p >> 8)) >> 8) as u8
}

/// A decomposed rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SplitRect {
    /// Main rectangle interior, or the complete rectangle when it is not split.
    pub(crate) main: RectPart,
    /// Top antialiased strip, if required.
    pub(crate) top: Option<RectPart>,
    /// Bottom antialiased strip, if required.
    pub(crate) bottom: Option<RectPart>,
    /// Left antialiased strip between the top and bottom strips, if required.
    pub(crate) left: Option<RectPart>,
    /// Right antialiased strip between the top and bottom strips, if required.
    pub(crate) right: Option<RectPart>,
}

/// Decompose `rect` into pixel-aligned parts, or return `None` if it covers no pixels.
///
/// Like the strip renderer, this works on the edges converted to `f32`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "edges are narrowed to f32 on purpose, like in the strip renderer, and recorded rect \
              coordinates are clipped to the u16 viewport domain before packing"
)]
pub(crate) fn split_rect(rect: &Rect) -> Option<SplitRect> {
    let x0 = rect.x0 as f32;
    let y0 = rect.y0 as f32;
    let x1 = rect.x1 as f32;
    let y1 = rect.y1 as f32;
    // Distinct edges can be converted to the same `f32`.
    if !(x0 < x1 && y0 < y1) {
        return None;
    }

    let sx0 = x0.floor();
    let sy0 = y0.floor();
    let sx1 = x1.ceil();
    let sy1 = y1.ceil();

    let x = sx0 as u16;
    let y = sy0 as u16;
    let width = (sx1 - sx0) as u16;
    let height = (sy1 - sy0) as u16;

    let part = |bounds: RectU16| RectPart::new(bounds, x0, y0, x1, y1);

    // There's a balance to strike between reducing work in the fragment shader by splitting
    // out the inner part of the rectangle without anti-aliasing, and additional overhead
    // that arises from rendering 5 rectangles instead of just one. While the exact threshold
    // will obviously depend on the device, some experiments on a low-tier tablet showed that
    // `LARGE_RECT_SPLIT_THRESHOLD` seems to be a a reasonable value.
    if x1 - x0 < f32::from(LARGE_RECT_SPLIT_THRESHOLD)
        || y1 - y0 < f32::from(LARGE_RECT_SPLIT_THRESHOLD)
    {
        return Some(SplitRect {
            main: part(RectU16::new(x, y, x + width, y + height)),
            top: None,
            bottom: None,
            left: None,
            right: None,
        });
    }

    let has_left_aa = x0 > sx0;
    let has_top_aa = y0 > sy0;
    let has_right_aa = x1 < sx1;
    let has_bottom_aa = y1 < sy1;
    let has_top_strip = has_top_aa || has_left_aa || has_right_aa;
    let has_bottom_strip = has_bottom_aa || has_left_aa || has_right_aa;
    let left_inset = u16::from(has_left_aa);
    let right_inset = u16::from(has_right_aa);
    let top_inset = u16::from(has_top_strip);
    let bottom_inset = u16::from(has_bottom_strip);
    let inner_x = x + left_inset;
    let inner_y = y + top_inset;
    // Can't underflow because rectangles have at least `LARGE_RECT_SPLIT_THRESHOLD` in each
    // direction, which is larger than 2.
    let inner_width = width - left_inset - right_inset;
    let inner_height = height - top_inset - bottom_inset;

    Some(SplitRect {
        main: RectPart::full(RectU16::new(
            inner_x,
            inner_y,
            inner_x + inner_width,
            inner_y + inner_height,
        )),
        top: has_top_strip.then(|| part(RectU16::new(x, y, x + width, y + 1))),
        bottom: has_bottom_strip
            .then(|| part(RectU16::new(x, y + height - 1, x + width, y + height))),
        left: has_left_aa.then(|| part(RectU16::new(x, inner_y, x + 1, inner_y + inner_height))),
        right: has_right_aa.then(|| {
            part(RectU16::new(
                x + width - 1,
                inner_y,
                x + width,
                inner_y + inner_height,
            ))
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::{RectPart, SplitRect, combine_alphas, split_rect};

    use vello_common::geometry::RectU16;
    use vello_common::kurbo::Rect;
    use vello_common::rect::{coverage_to_alpha, pixel_coverage};

    fn part(x: u16, y: u16, width: u16, height: u16, alphas: [u8; 4]) -> RectPart {
        RectPart {
            rect: RectU16::new(x, y, x + width, y + height),
            alphas: u32::from_le_bytes(alphas),
            corner_corrections: 0,
        }
    }

    fn full_part(x: u16, y: u16, width: u16, height: u16) -> RectPart {
        RectPart::full(RectU16::new(x, y, x + width, y + height))
    }

    #[test]
    fn splitter_keeps_small_rect_whole() {
        let rect = Rect::new(10.25, 20.5, 25.75, 35.25);
        let split = split_rect(&rect).unwrap();

        assert_eq!(
            split,
            SplitRect {
                main: part(10, 20, 16, 16, [191, 191, 128, 64]),
                top: None,
                bottom: None,
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_keeps_subpixel_rect_inside_one_pixel() {
        let rect = Rect::new(10.125, 20.25, 10.875, 20.75);
        let split = split_rect(&rect).unwrap();

        assert_eq!(
            split,
            SplitRect {
                main: part(10, 20, 1, 1, [191, 191, 128, 128]),
                top: None,
                bottom: None,
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_keeps_subpixel_rect_spanning_two_pixels_in_width() {
        let rect = Rect::new(10.75, 20.125, 11.25, 20.875);
        let split = split_rect(&rect).unwrap();

        assert_eq!(
            split,
            SplitRect {
                main: part(10, 20, 2, 1, [64, 64, 191, 191]),
                top: None,
                bottom: None,
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_keeps_subpixel_rect_spanning_two_pixels_in_height() {
        let rect = Rect::new(10.125, 20.75, 10.875, 21.25);
        let split = split_rect(&rect).unwrap();

        assert_eq!(
            split,
            SplitRect {
                main: part(10, 20, 1, 2, [191, 191, 64, 64]),
                top: None,
                bottom: None,
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_keeps_multi_pixel_width_rect_within_one_pixel_height() {
        let rect = Rect::new(10.25, 20.125, 14.75, 20.875);
        let split = split_rect(&rect).unwrap();

        assert_eq!(
            split,
            SplitRect {
                main: part(10, 20, 5, 1, [191, 191, 191, 191]),
                top: None,
                bottom: None,
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_keeps_multi_pixel_height_rect_within_one_pixel_width() {
        let rect = Rect::new(10.125, 20.25, 10.875, 24.75);
        let split = split_rect(&rect).unwrap();

        assert_eq!(
            split,
            SplitRect {
                main: part(10, 20, 1, 5, [191, 191, 191, 191]),
                top: None,
                bottom: None,
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_splits_large_rect_into_five_parts() {
        let rect = Rect::new(10.25, 20.5, 42.75, 52.75);
        let split = split_rect(&rect).unwrap();

        assert_eq!(
            split,
            SplitRect {
                main: full_part(11, 21, 31, 31),
                top: Some(part(10, 20, 33, 1, [191, 191, 128, 128])),
                bottom: Some(part(10, 52, 33, 1, [191, 191, 191, 191])),
                left: Some(part(10, 21, 1, 31, [191, 191, 255, 255])),
                right: Some(part(42, 21, 1, 31, [191, 191, 255, 255])),
            }
        );
    }

    #[test]
    fn splitter_omits_unneeded_edge_parts() {
        let rect = Rect::new(10.0, 20.5, 42.0, 53.0);
        let split = split_rect(&rect).unwrap();

        assert_eq!(
            split,
            SplitRect {
                main: full_part(10, 21, 32, 32),
                top: Some(part(10, 20, 32, 1, [255, 255, 128, 128])),
                bottom: None,
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_handles_large_rect_with_only_vertical_aa() {
        let rect = Rect::new(5.0, 2.25, 37.0, 34.75);
        let split = split_rect(&rect).unwrap();

        assert_eq!(
            split,
            SplitRect {
                main: full_part(5, 3, 32, 31),
                top: Some(part(5, 2, 32, 1, [255, 255, 191, 191])),
                bottom: Some(part(5, 34, 32, 1, [255, 255, 191, 191])),
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_keeps_large_aligned_rect_as_single_main_rect() {
        let rect = Rect::new(10.0, 20.0, 42.0, 60.0);
        let split = split_rect(&rect).unwrap();

        assert_eq!(
            split,
            SplitRect {
                main: full_part(10, 20, 32, 40),
                top: None,
                bottom: None,
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_skips_rect_without_pixels_in_f32() {
        // Both horizontal edges are converted to the same `f32`, between 4300 and 4301.
        let rect = Rect::new(4300.3001, 1.0, 4300.3002, 2.0);
        assert_eq!(split_rect(&rect), None);
    }

    #[test]
    fn near_integer_edges_keep_corner_corrections() {
        // All boundary alphas round to 255, but the exact top-left alpha rounds to 254.
        let split = split_rect(&Rect::new(10.001, 20.001, 30.0, 40.0)).unwrap();
        assert_eq!(split.main.alphas, u32::MAX);
        assert_eq!(split.main.corner_corrections, 0b11);
        assert!(!split.main.is_full());

        let split = split_rect(&Rect::new(10.0, 20.0, 30.0, 40.0)).unwrap();
        assert!(split.main.is_full());
    }

    /// Sweep sub-pixel offsets and sizes. At every corner of every part, the exact alpha must be
    /// within one of the combined column and row alphas, and the corner correction must restore
    /// it.
    #[test]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "mirrors the f32 conversion of split_rect"
    )]
    fn corner_corrections_reproduce_exact_alphas() {
        let mut corners = 0;
        let mut full_corners = 0;
        // Besides a 1/64 grid, offsets within half an alpha step of the pixel grid, which need
        // a correction even though their boundary alphas are 255.
        let offsets = (0..64)
            .map(|i| f64::from(i) / 64.0)
            .chain([0.0005, 0.001, 0.0019]);
        for dx in offsets.clone() {
            for dy in offsets.clone() {
                for (width, height) in [
                    (0.4, 0.7),
                    (3.3, 2.6),
                    (9.5, 1.2),
                    (29.999, 19.999),
                    (40.7, 35.2),
                ] {
                    let rect = Rect::new(7.0 + dx, 11.0 + dy, 7.0 + dx + width, 11.0 + dy + height);
                    let split = split_rect(&rect).unwrap();
                    let [x0, y0, x1, y1] = [rect.x0, rect.y0, rect.x1, rect.y1].map(|v| v as f32);
                    let edges = [split.top, split.bottom, split.left, split.right];
                    for part in edges.into_iter().flatten().chain([split.main]) {
                        let alphas = part.alphas.to_le_bytes();
                        let columns = [part.rect.x0, part.rect.x1 - 1];
                        let rows = [part.rect.y0, part.rect.y1 - 1];
                        for (row, py) in rows.into_iter().enumerate() {
                            for (col, px) in columns.into_iter().enumerate() {
                                let cov_x = pixel_coverage(f32::from(px), x0, x1);
                                let cov_y = pixel_coverage(f32::from(py), y0, y1);
                                let exact = coverage_to_alpha(cov_x * cov_y);
                                let combined = combine_alphas(alphas[col], alphas[2 + row]);
                                assert!(
                                    exact.abs_diff(combined) <= 1,
                                    "pixel ({px}, {py}) of {part:?} in {rect:?}"
                                );

                                // Apply the correction like the shader does.
                                let correction =
                                    u32::from(part.corner_corrections >> (2 * (col + 2 * row)));
                                let alpha = u32::from(combined) + ((correction + 1) & 0b11) - 1;
                                assert_eq!(
                                    alpha,
                                    u32::from(exact),
                                    "pixel ({px}, {py}) of {part:?} in {rect:?}"
                                );

                                corners += 1;
                                if part.is_full() {
                                    assert_eq!(exact, 255, "pixel ({px}, {py}) of {rect:?}");
                                    full_corners += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(corners > 100_000, "only {corners} corners checked");
        assert!(
            full_corners > 100,
            "only {full_corners} full corners checked"
        );
    }
}
