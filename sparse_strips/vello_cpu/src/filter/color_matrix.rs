// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Color matrix filter implementation.

use super::FilterEffect;
use super::pixel::{clamp_unit, premul_rgba8_to_straight_f32, straight_f32_to_premul_rgba8};
use crate::layer_manager::LayerManager;
use vello_common::filter::color_matrix::ColorMatrix;
use vello_common::peniko::color::PremulRgba8;
use vello_common::pixmap::Pixmap;

impl FilterEffect for ColorMatrix {
    fn execute_lowp(&self, pixmap: &mut Pixmap, _layer_manager: &mut LayerManager) {
        apply_color_matrix(pixmap, &self.matrix);
    }

    fn execute_highp(&self, pixmap: &mut Pixmap, _layer_manager: &mut LayerManager) {
        apply_color_matrix(pixmap, &self.matrix);
    }
}

fn apply_color_matrix(pixmap: &mut Pixmap, matrix: &[f32; 20]) {
    for pixel in pixmap.data_mut() {
        *pixel = apply_color_matrix_to_pixel(*pixel, matrix);
    }

    pixmap.recompute_may_have_transparency();
}

fn apply_color_matrix_to_pixel(pixel: PremulRgba8, matrix: &[f32; 20]) -> PremulRgba8 {
    let [r, g, b, a] = premul_rgba8_to_straight_f32(pixel);

    let out_r = apply_row(matrix, 0, r, g, b, a);
    let out_g = apply_row(matrix, 5, r, g, b, a);
    let out_b = apply_row(matrix, 10, r, g, b, a);
    let out_a = apply_row(matrix, 15, r, g, b, a);

    straight_f32_to_premul_rgba8(out_r, out_g, out_b, out_a)
}

fn apply_row(matrix: &[f32; 20], offset: usize, r: f32, g: f32, b: f32, a: f32) -> f32 {
    clamp_unit(
        matrix[offset] * r
            + matrix[offset + 1] * g
            + matrix[offset + 2] * b
            + matrix[offset + 3] * a
            + matrix[offset + 4],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vello_common::filter_effects::matrices;

    #[test]
    fn identity_preserves_premultiplied_pixel() {
        let pixel = PremulRgba8 {
            r: 128,
            g: 64,
            b: 32,
            a: 128,
        };

        assert_eq!(
            apply_color_matrix_to_pixel(pixel, &matrices::IDENTITY),
            pixel
        );
    }

    #[test]
    fn grayscale_uses_unpremultiplied_color_channels() {
        let pixel = PremulRgba8 {
            r: 128,
            g: 0,
            b: 0,
            a: 128,
        };

        assert_eq!(
            apply_color_matrix_to_pixel(pixel, &matrices::GRAYSCALE),
            PremulRgba8 {
                r: 27,
                g: 27,
                b: 27,
                a: 128,
            }
        );
    }

    #[test]
    fn matrix_offsets_can_create_color_from_transparent_black() {
        let matrix = [
            0.0, 0.0, 0.0, 0.0, 0.5, //
            0.0, 0.0, 0.0, 0.0, 0.25, //
            0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.5,
        ];

        assert_eq!(
            apply_color_matrix_to_pixel(PremulRgba8::from_u32(0), &matrix),
            PremulRgba8 {
                r: 64,
                g: 32,
                b: 0,
                a: 128,
            }
        );
    }

    #[test]
    fn execution_updates_pixmap_transparency_flag() {
        let mut pixmap = Pixmap::from_parts_with_opacity(
            alloc::vec![PremulRgba8 {
                r: 255,
                g: 0,
                b: 0,
                a: 255,
            }],
            1,
            1,
            false,
        );
        let mut layer_manager = LayerManager::new();
        let matrix = [
            1.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.5,
        ];
        let filter = ColorMatrix::new(matrix);

        filter.execute_lowp(&mut pixmap, &mut layer_manager);

        assert!(pixmap.may_have_transparency());
        assert_eq!(
            pixmap.sample(0, 0),
            PremulRgba8 {
                r: 128,
                g: 0,
                b: 0,
                a: 128,
            }
        );
    }
}
