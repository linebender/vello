// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `feColorMatrix` filter primitive implementation.

use vello_common::filter::color_matrix::ColorMatrix;
use vello_common::peniko::color::PremulRgba8;
use vello_common::pixmap::Pixmap;

use super::FilterEffect;
use crate::filter::context::ScratchBuffer;

impl FilterEffect for ColorMatrix {
    fn execute_lowp(&self, pixmap: &mut Pixmap, _: &mut ScratchBuffer) {
        apply_color_matrix(pixmap, &self.matrix);
    }

    fn execute_highp(&self, pixmap: &mut Pixmap, _: &mut ScratchBuffer) {
        apply_color_matrix(pixmap, &self.matrix);
    }
}

// TODO: Use SIMD once filters are wired up for it. Being a per-pixel filter, this could also
// skip the spatial filter path (and its intermediate pixmap) and run directly in fine
// rasterization, as could `Flood`.
fn apply_color_matrix(pixmap: &mut Pixmap, matrix: &[f32; 20]) {
    if is_premul_compatible(matrix) {
        map_pixels(pixmap, |pixel| apply_premul(pixel, matrix));
    } else {
        map_pixels(pixmap, |pixel| apply_straight(pixel, matrix));
    }
}

#[inline]
fn map_pixels(pixmap: &mut Pixmap, f: impl Fn(PremulRgba8) -> PremulRgba8) {
    for pixel in pixmap.data_mut() {
        *pixel = f(*pixel);
    }
}

/// Whether applying the matrix directly to premultiplied channels gives the same result as
/// unpremultiplying, applying it, and premultiplying again.
///
/// This holds if the color rows neither read alpha nor add an offset, and alpha is preserved.
#[inline]
fn is_premul_compatible(matrix: &[f32; 20]) -> bool {
    let row_r = &matrix[0..5];
    let row_g = &matrix[5..10];
    let row_b = &matrix[10..15];
    let row_a = &matrix[15..20];
    let color_rows_alpha_independent = [row_r, row_g, row_b]
        .iter()
        .all(|row| row[3] == 0.0 && row[4] == 0.0);
    let alpha_preserved = row_a == [0.0, 0.0, 0.0, 1.0, 0.0];

    color_rows_alpha_independent && alpha_preserved
}

/// Apply the matrix to the straight-alpha color of a pixel.
#[inline]
fn apply_straight(pixel: PremulRgba8, matrix: &[f32; 20]) -> PremulRgba8 {
    let a = u8_to_norm(pixel.a);
    // The color of a fully transparent pixel is undefined; treat it as transparent black.
    let inv_a = if pixel.a == 0 { 0.0 } else { 1.0 / a };
    let [r, g, b] = [pixel.r, pixel.g, pixel.b].map(|c| u8_to_norm(c) * inv_a);
    let row = |i: usize| {
        (matrix[i] * r + matrix[i + 1] * g + matrix[i + 2] * b + matrix[i + 3] * a + matrix[i + 4])
            .clamp(0.0, 1.0)
    };
    let out_a = row(15);

    PremulRgba8 {
        r: round_u8(row(0) * out_a * 255.0),
        g: round_u8(row(5) * out_a * 255.0),
        b: round_u8(row(10) * out_a * 255.0),
        a: round_u8(out_a * 255.0),
    }
}

/// Apply the color rows of a matrix that satisfies [`is_premul_compatible`] directly to the
/// premultiplied channels.
#[inline]
fn apply_premul(pixel: PremulRgba8, matrix: &[f32; 20]) -> PremulRgba8 {
    // The matrix is linear in the color channels, so it can be applied in [0, 255] units.
    let [r, g, b, a] = [pixel.r, pixel.g, pixel.b, pixel.a].map(f32::from);
    // Clamping the straight color to [0, 1] is clamping the premultiplied one to [0, a].
    let row = |i: usize| (matrix[i] * r + matrix[i + 1] * g + matrix[i + 2] * b).clamp(0.0, a);

    PremulRgba8 {
        r: round_u8(row(0)),
        g: round_u8(row(5)),
        b: round_u8(row(10)),
        a: pixel.a,
    }
}

#[inline]
fn u8_to_norm(value: u8) -> f32 {
    f32::from(value) * (1.0 / 255.0)
}

/// Round a value in `[0, 255]` to the nearest integer.
#[inline]
fn round_u8(value: f32) -> u8 {
    // Truncating after adding 0.5 rounds non-negative values, without needing `f32::round`.
    (value + 0.5) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use vello_common::color::{AlphaColor, Srgb};
    use vello_common::filter_effects::matrices;

    const PREMUL_PIXEL: PremulRgba8 = PremulRgba8 {
        r: 80,
        g: 32,
        b: 16,
        a: 128,
    };

    #[test]
    fn identity_preserves_premultiplied_pixel() {
        assert_eq!(
            apply_straight(PREMUL_PIXEL, &matrices::IDENTITY),
            PREMUL_PIXEL
        );
    }

    #[test]
    fn grayscale_uses_straight_alpha_color() {
        let pixel = PremulRgba8 {
            r: 128,
            g: 0,
            b: 0,
            a: 128,
        };

        assert_eq!(
            apply_straight(pixel, &matrices::GRAYSCALE),
            PremulRgba8 {
                r: 27,
                g: 27,
                b: 27,
                a: 128,
            }
        );
    }

    #[test]
    fn premul_compatibility() {
        assert!(is_premul_compatible(&matrices::GRAYSCALE));
        assert!(is_premul_compatible(&matrices::SEPIA));
        assert!(!is_premul_compatible(&matrices::ALPHA_TO_BLACK));

        let mut with_color_offset = matrices::SEPIA;
        with_color_offset[9] = 0.1;
        assert!(!is_premul_compatible(&with_color_offset));

        let mut with_alpha_scale = matrices::SEPIA;
        with_alpha_scale[18] = 0.5;
        assert!(!is_premul_compatible(&with_alpha_scale));
    }

    #[test]
    fn premul_path_matches_straight_path() {
        for matrix in [matrices::SEPIA, matrices::GRAYSCALE] {
            for a in [1, 128, 255] {
                for [r, g, b] in [[255, 0, 0], [80, 160, 240], [255, 255, 255]] {
                    let pixel = AlphaColor::<Srgb>::from_rgba8(r, g, b, a)
                        .premultiply()
                        .to_rgba8();
                    assert_eq!(
                        apply_premul(pixel, &matrix),
                        apply_straight(pixel, &matrix),
                        "{pixel:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn premul_path_clamps_color_to_alpha() {
        let pixel = PremulRgba8 {
            r: 128,
            g: 0,
            b: 0,
            a: 128,
        };
        let matrix = [
            2.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, 0.0,
        ];

        assert_eq!(apply_premul(pixel, &matrix), pixel);
        assert_eq!(apply_straight(pixel, &matrix), pixel);
    }

    #[test]
    fn offsets_add_color_to_transparent_black() {
        let matrix = [
            0.0, 0.0, 0.0, 0.0, 0.5, //
            0.0, 0.0, 0.0, 0.0, 0.25, //
            0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.5,
        ];

        assert_eq!(
            apply_straight(PremulRgba8::from_u32(0), &matrix),
            PremulRgba8 {
                r: 64,
                g: 32,
                b: 0,
                a: 128,
            }
        );
    }
}
