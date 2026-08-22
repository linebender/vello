// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Helpers for decomposing rectangles used by the rectangle fast path.

use vello_common::geometry::RectU16;
use vello_common::kurbo::Rect;
use vello_common::rect::{combine_coverage_u8, corner_coverage_u8, coverage_to_u8, pixel_coverage};

/// The threshold of the rectangle size after which a rectangle should be split up
/// into multiple smaller ones.
const LARGE_RECT_SPLIT_THRESHOLD: u16 = 32;

/// The packed coverage value of a part whose boundary bytes are all 255.
///
/// This alone does not mean every pixel is exactly 255: an edge less than half
/// an alpha step inside an integer boundary still rounds its axis byte to 255
/// while the exact corner alpha is 254. Only together with
/// [`FULLY_OPAQUE_ADJUST`] does it mark a part the shader may skip entirely.
pub(crate) const FULL_COVERAGE: u32 = 0xFFFF_FFFF;

/// The `corner_adjust` marker for a part whose pixels are all exactly 255.
///
/// Real corrections store `correction + 1` per 2-bit slot, so every slot is at
/// most 2 and `0xFF` is unreachable — it is free to act as the "no
/// anti-aliasing work at all" flag. Such parts skip the shader's AA branch and
/// are eligible for the opaque (depth-tested) pass when their paint is opaque.
pub(crate) const FULLY_OPAQUE_ADJUST: u8 = 0xFF;

/// Integer rectangle geometry and its packed per-pixel boundary coverage.
///
/// The shader reconstructs each pixel's alpha from `frac` exactly as the CPU strip renderer
/// (`vello_common::rect`) would have written it: boundary rows/columns use their coverage byte
/// directly, and corner pixels combine the two axis bytes with `round(x * y / 255)` plus the
/// matching 2-bit correction from `corner_adjust`. This keeps a rect drawn as a GPU quad
/// byte-identical to the same rect drawn as strips (e.g. when a clip forces the strip path).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RectPart {
    /// Pixel-aligned bounds of this rectangle part.
    pub(crate) rect: RectU16,
    /// Coverage bytes of the part's boundary pixels: first column, last column, first row,
    /// last row (low byte to high byte). Interior pixels are always fully covered.
    pub(crate) frac: u32,
    /// Corner corrections, 2 bits per corner, storing `correction + 1` where `correction` is
    /// the difference between the exact corner alpha and the shader's integer
    /// `round(x * y / 255)` of the two axis bytes (always -1, 0, or 1). Corner slot index is
    /// `is_last_column | (is_last_row << 1)`. [`FULLY_OPAQUE_ADJUST`] when every pixel of the
    /// part is exactly 255.
    pub(crate) corner_adjust: u8,
}

impl RectPart {
    pub(crate) fn shift(self, shift: (i32, i32)) -> Self {
        Self {
            rect: self.rect.shift(shift),
            ..self
        }
    }
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

/// Split `rect` into pixel-aligned parts with packed boundary coverage, or `None` when the
/// rectangle covers no pixels after `f32` conversion.
///
/// The edges are converted to `f32` before anything else, in the same order as the CPU strip
/// renderer (`vello_common::rect::render`), and the coverage bytes come from the same per-pixel
/// formula — so the quads drawn from these parts produce exactly the bytes strips would.
#[expect(
    clippy::cast_possible_truncation,
    reason = "recorded rect coordinates are clipped to the u16 viewport domain before packing"
)]
pub(crate) fn split_rect(rect: &Rect) -> Option<SplitRect> {
    let rect_x0 = rect.x0 as f32;
    let rect_y0 = rect.y0 as f32;
    let rect_x1 = rect.x1 as f32;
    let rect_y1 = rect.y1 as f32;

    let sx0 = rect_x0.floor();
    let sy0 = rect_y0.floor();
    let sx1 = rect_x1.ceil();
    let sy1 = rect_y1.ceil();

    let x = sx0 as u16;
    let y = sy0 as u16;
    // Are guaranteed to be >= 0 since we rejected negative rectangles.
    let width = (sx1 - sx0) as u16;
    let height = (sy1 - sy0) as u16;

    if width == 0 || height == 0 {
        return None;
    }

    let part = |part_rect: RectU16| -> RectPart {
        let cov_x0 = pixel_coverage(f32::from(part_rect.x0), rect_x0, rect_x1);
        let cov_x1 = pixel_coverage(f32::from(part_rect.x1 - 1), rect_x0, rect_x1);
        let cov_y0 = pixel_coverage(f32::from(part_rect.y0), rect_y0, rect_y1);
        let cov_y1 = pixel_coverage(f32::from(part_rect.y1 - 1), rect_y0, rect_y1);
        let bx0 = coverage_to_u8(cov_x0);
        let bx1 = coverage_to_u8(cov_x1);
        let by0 = coverage_to_u8(cov_y0);
        let by1 = coverage_to_u8(cov_y1);

        let frac = u32::from(bx0)
            | (u32::from(bx1) << 8)
            | (u32::from(by0) << 16)
            | (u32::from(by1) << 24);

        let adjust = |cov_x: f32, cov_y: f32, bx: u8, by: u8| -> u8 {
            let correction = i16::from(corner_coverage_u8(cov_x, cov_y))
                - i16::from(combine_coverage_u8(bx, by));
            debug_assert!(
                (-1..=1).contains(&correction),
                "corner correction out of range: {correction}"
            );
            (correction + 1) as u8
        };
        let corner_adjust = adjust(cov_x0, cov_y0, bx0, by0)
            | (adjust(cov_x1, cov_y0, bx1, by0) << 2)
            | (adjust(cov_x0, cov_y1, bx0, by1) << 4)
            | (adjust(cov_x1, cov_y1, bx1, by1) << 6);

        // All-255 boundary bytes do not by themselves make the part fully
        // opaque: an edge less than half an alpha step inside an integer
        // rounds its byte to 255 while the exact corner alpha is 254 (a -1
        // correction). Only when the corrections are all zero too may the
        // shader skip the part's anti-aliasing work entirely.
        let corner_adjust = if frac == FULL_COVERAGE && corner_adjust == 0b0101_0101 {
            FULLY_OPAQUE_ADJUST
        } else {
            corner_adjust
        };

        RectPart {
            rect: part_rect,
            frac,
            corner_adjust,
        }
    };

    // There's a balance to strike between reducing work in the fragment shader by splitting
    // out the inner part of the rectangle without anti-aliasing, and additional overhead
    // that arises from rendering 5 rectangles instead of just one. While the exact threshold
    // will obviously depend on the device, some experiments on a low-tier tablet showed that
    // `LARGE_RECT_SPLIT_THRESHOLD` seems to be a a reasonable value.
    if rect.x1 - rect.x0 < f64::from(LARGE_RECT_SPLIT_THRESHOLD)
        || rect.y1 - rect.y0 < f64::from(LARGE_RECT_SPLIT_THRESHOLD)
    {
        return Some(SplitRect {
            main: part(RectU16::new(x, y, x + width, y + height)),
            top: None,
            bottom: None,
            left: None,
            right: None,
        });
    }

    let has_left_aa = rect_x0 > sx0;
    let has_top_aa = rect_y0 > sy0;
    let has_right_aa = rect_x1 < sx1;
    let has_bottom_aa = rect_y1 < sy1;
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
        main: RectPart {
            rect: RectU16::new(
                inner_x,
                inner_y,
                inner_x + inner_width,
                inner_y + inner_height,
            ),
            frac: FULL_COVERAGE,
            corner_adjust: FULLY_OPAQUE_ADJUST,
        },
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
    use super::{FULL_COVERAGE, FULLY_OPAQUE_ADJUST, RectPart, SplitRect, split_rect};

    use vello_common::geometry::RectU16;
    use vello_common::kurbo::Rect;
    use vello_common::rect::{combine_coverage_u8, corner_coverage_u8, coverage_to_u8};

    /// The all-zero-correction encoding: every corner stores `0 + 1`.
    const NO_ADJUST: u8 = 0b0101_0101;

    fn part(x: u16, y: u16, width: u16, height: u16, frac: [u8; 4], corner_adjust: u8) -> RectPart {
        RectPart {
            rect: RectU16::new(x, y, x + width, y + height),
            frac: u32::from(frac[0])
                | (u32::from(frac[1]) << 8)
                | (u32::from(frac[2]) << 16)
                | (u32::from(frac[3]) << 24),
            corner_adjust,
        }
    }

    fn full_part(x: u16, y: u16, width: u16, height: u16) -> RectPart {
        RectPart {
            rect: RectU16::new(x, y, x + width, y + height),
            frac: FULL_COVERAGE,
            corner_adjust: FULLY_OPAQUE_ADJUST,
        }
    }

    #[test]
    fn splitter_keeps_small_rect_whole() {
        let rect = Rect::new(10.25, 20.5, 25.75, 35.25);
        let split = split_rect(&rect).unwrap();

        // Boundary pixel coverages: left column [10.25, 11) -> 0.75, right column
        // [25, 25.75) -> 0.75, top row [20.5, 21) -> 0.5, bottom row [35, 35.25) -> 0.25.
        assert_eq!(
            split,
            SplitRect {
                main: part(10, 20, 16, 16, [191, 191, 128, 64], NO_ADJUST),
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

        // A single pixel holds both edges of each axis: x coverage 0.75, y coverage 0.5,
        // stored in both the first and last slot.
        assert_eq!(
            split,
            SplitRect {
                main: part(10, 20, 1, 1, [191, 191, 128, 128], NO_ADJUST),
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
                main: part(10, 20, 2, 1, [64, 64, 191, 191], NO_ADJUST),
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
                main: part(10, 20, 1, 2, [191, 191, 64, 64], NO_ADJUST),
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
                main: part(10, 20, 5, 1, [191, 191, 191, 191], NO_ADJUST),
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
                main: part(10, 20, 1, 5, [191, 191, 191, 191], NO_ADJUST),
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

        // Boundary coverages: left 0.75 -> 191, right 0.75 -> 191, top 0.5 -> 128,
        // bottom 0.75 -> 191. A 1-pixel-thick strip has the same pixel as first and last
        // on its thin axis, so that byte repeats in both slots.
        assert_eq!(
            split,
            SplitRect {
                main: full_part(11, 21, 31, 31),
                top: Some(part(10, 20, 33, 1, [191, 191, 128, 128], NO_ADJUST)),
                bottom: Some(part(10, 52, 33, 1, [191, 191, 191, 191], NO_ADJUST)),
                left: Some(part(10, 21, 1, 31, [191, 191, 255, 255], NO_ADJUST)),
                right: Some(part(42, 21, 1, 31, [191, 191, 255, 255], NO_ADJUST)),
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
                top: Some(part(10, 20, 32, 1, [255, 255, 128, 128], NO_ADJUST)),
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
                top: Some(part(5, 2, 32, 1, [255, 255, 191, 191], NO_ADJUST)),
                bottom: Some(part(5, 34, 32, 1, [255, 255, 191, 191], NO_ADJUST)),
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
    fn splitter_rejects_rect_collapsing_to_zero_pixels() {
        assert_eq!(split_rect(&Rect::new(10.0, 20.0, 10.0, 25.0)), None);
    }

    /// Sweep sub-pixel offsets and sizes and check that, for every part, applying the packed
    /// correction to the shader's integer combine of the packed bytes reproduces the exact
    /// corner byte the CPU strip renderer writes (`corner_coverage_u8`).
    #[test]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "mirrors split_rect's f32 conversion and reconstructs a u8 alpha"
    )]
    fn corner_corrections_reproduce_exact_corner_bytes() {
        let mut checked = 0_u64;
        let mut full_seen = 0_u64;
        // The 1/64 grid plus offsets inside (0, 1/510]: edges that close to an
        // integer round their axis byte to 255 while the exact corner alpha is
        // 254, which is precisely the case the fully-opaque marker must not
        // swallow.
        let offsets = || {
            (0..64_u32)
                .map(|i| f64::from(i) / 64.0)
                .chain([0.0005, 0.001, 0.0019])
        };
        for dx in offsets() {
            for dy in offsets() {
                for (w, h) in [
                    (0.4, 0.7),
                    (3.3, 2.6),
                    (9.5, 1.2),
                    (29.999, 19.999),
                    (40.7, 35.2),
                ] {
                    let rect = Rect::new(7.0 + dx, 11.0 + dy, 7.0 + dx + w, 11.0 + dy + h);
                    let Some(split) = split_rect(&rect) else {
                        continue;
                    };
                    let rect_x0 = rect.x0 as f32;
                    let rect_y0 = rect.y0 as f32;
                    let rect_x1 = rect.x1 as f32;
                    let rect_y1 = rect.y1 as f32;
                    for part in [
                        Some(split.main),
                        split.top,
                        split.bottom,
                        split.left,
                        split.right,
                    ]
                    .into_iter()
                    .flatten()
                    {
                        let bytes = part.frac.to_le_bytes();
                        for (slot, (px, py)) in [
                            (part.rect.x0, part.rect.y0),
                            (part.rect.x1 - 1, part.rect.y0),
                            (part.rect.x0, part.rect.y1 - 1),
                            (part.rect.x1 - 1, part.rect.y1 - 1),
                        ]
                        .into_iter()
                        .enumerate()
                        {
                            let cov_x = super::pixel_coverage(f32::from(px), rect_x0, rect_x1);
                            let cov_y = super::pixel_coverage(f32::from(py), rect_y0, rect_y1);
                            let exact = corner_coverage_u8(cov_x, cov_y);
                            assert_eq!(coverage_to_u8(cov_x), bytes[slot & 1]);
                            assert_eq!(coverage_to_u8(cov_y), bytes[2 + (slot >> 1)]);
                            if part.corner_adjust == FULLY_OPAQUE_ADJUST {
                                // The fully-opaque marker asserts every pixel
                                // is exactly 255 — including the corners.
                                assert_eq!(
                                    exact, 255,
                                    "fully-opaque part with a non-255 corner for {rect:?} slot {slot}"
                                );
                                full_seen += 1;
                            } else {
                                let combined =
                                    combine_coverage_u8(bytes[slot & 1], bytes[2 + (slot >> 1)]);
                                let adj = (part.corner_adjust >> (slot * 2)) & 0x3;
                                let reconstructed =
                                    (i16::from(combined) + i16::from(adj) - 1) as u8;
                                assert_eq!(
                                    reconstructed, exact,
                                    "corner mismatch for {rect:?} part {part:?} slot {slot}"
                                );
                            }
                            checked += 1;
                        }
                    }
                }
            }
        }
        // Make sure the sweep exercised a meaningful number of corners on
        // both sides of the fully-opaque split.
        assert!(checked > 100_000, "only checked {checked} corners");
        assert!(full_seen > 100, "only saw {full_seen} fully-opaque corners");
    }

    /// The near-integer regression: all four boundary bytes quantize to 255,
    /// but the exact corner alpha is 254 — the part must carry the -1
    /// corrections instead of the fully-opaque marker.
    #[test]
    fn near_integer_edges_keep_their_corner_corrections() {
        let split = split_rect(&Rect::new(10.001, 20.001, 30.0, 40.0)).unwrap();
        let part = split.main;
        assert_eq!(part.frac, FULL_COVERAGE);
        assert_ne!(part.corner_adjust, FULLY_OPAQUE_ADJUST);
        // Top-left corner (slot 0): both axis coverages just under 1.0, the
        // exact alpha is 254, combine(255, 255) is 255, so the correction is
        // -1 (encoded 0); the other three corners are exact (encoded 1).
        assert_eq!(part.corner_adjust, 0b0101_0100);

        // A genuinely integer rectangle is fully opaque.
        let split = split_rect(&Rect::new(10.0, 20.0, 30.0, 40.0)).unwrap();
        assert_eq!(split.main.corner_adjust, FULLY_OPAQUE_ADJUST);
    }
}
