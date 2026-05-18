// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Color matrix filter implementation.

use super::FilterEffect;
use super::pixel::{
    clamp_unit, norm_to_u8, premul_rgba8_to_straight_f32, straight_f32_to_premul_rgba8, u8_to_norm,
};
use crate::layer_manager::LayerManager;
use vello_common::filter::color_matrix::ColorMatrix;
use vello_common::filter_effects::matrices;
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
    if matrix == &matrices::IDENTITY {
        return;
    }

    let mut may_have_transparency = false;
    if is_premul_compatible_matrix(matrix) {
        // For RGB-only, alpha-preserving matrices, applying the RGB matrix to
        // premultiplied channels is equivalent to unpremultiply -> matrix ->
        // premultiply. This avoids the per-pixel alpha division.
        for pixel in pixmap.data_mut() {
            let transformed = apply_premul_color_matrix_to_pixel(*pixel, matrix);
            may_have_transparency |= transformed.a != 255;
            *pixel = transformed;
        }
    } else {
        for pixel in pixmap.data_mut() {
            let transformed = apply_color_matrix_to_pixel(*pixel, matrix);
            may_have_transparency |= transformed.a != 255;
            *pixel = transformed;
        }
    }

    pixmap.set_may_have_transparency(may_have_transparency);
}

fn apply_color_matrix_to_pixel(pixel: PremulRgba8, matrix: &[f32; 20]) -> PremulRgba8 {
    let [r, g, b, a] = premul_rgba8_to_straight_f32(pixel);

    let out_r = apply_row(matrix, 0, r, g, b, a);
    let out_g = apply_row(matrix, 5, r, g, b, a);
    let out_b = apply_row(matrix, 10, r, g, b, a);
    let out_a = apply_row(matrix, 15, r, g, b, a);

    straight_f32_to_premul_rgba8(out_r, out_g, out_b, out_a)
}

fn apply_premul_color_matrix_to_pixel(pixel: PremulRgba8, matrix: &[f32; 20]) -> PremulRgba8 {
    let r = u8_to_norm(pixel.r);
    let g = u8_to_norm(pixel.g);
    let b = u8_to_norm(pixel.b);
    let a = u8_to_norm(pixel.a);

    let out_r = matrix[0] * r + matrix[1] * g + matrix[2] * b;
    let out_g = matrix[5] * r + matrix[6] * g + matrix[7] * b;
    let out_b = matrix[10] * r + matrix[11] * g + matrix[12] * b;

    PremulRgba8 {
        // Straight-alpha clamping before re-premultiplication becomes
        // clamping to [0, alpha] in premultiplied space.
        r: norm_to_u8(out_r.clamp(0.0, a)),
        g: norm_to_u8(out_g.clamp(0.0, a)),
        b: norm_to_u8(out_b.clamp(0.0, a)),
        a: pixel.a,
    }
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

fn is_premul_compatible_matrix(matrix: &[f32; 20]) -> bool {
    matrix[3] == 0.0
        && matrix[4] == 0.0
        && matrix[8] == 0.0
        && matrix[9] == 0.0
        && matrix[13] == 0.0
        && matrix[14] == 0.0
        && matrix[15] == 0.0
        && matrix[16] == 0.0
        && matrix[17] == 0.0
        && matrix[18] == 1.0
        && matrix[19] == 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn identity_matrix_leaves_pixmap_unchanged() {
        let mut pixmap = Pixmap::from_parts_with_opacity(
            alloc::vec![
                PremulRgba8 {
                    r: 255,
                    g: 0,
                    b: 0,
                    a: 255,
                },
                PremulRgba8 {
                    r: 128,
                    g: 64,
                    b: 32,
                    a: 128,
                },
            ],
            2,
            1,
            true,
        );
        let before = [pixmap.sample(0, 0), pixmap.sample(1, 0)];

        apply_color_matrix(&mut pixmap, &matrices::IDENTITY);

        assert_eq!([pixmap.sample(0, 0), pixmap.sample(1, 0)], before);
        assert!(pixmap.may_have_transparency());
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
    fn premul_compatible_matrices_skip_straight_alpha_conversion() {
        assert!(is_premul_compatible_matrix(&matrices::GRAYSCALE));
        assert!(is_premul_compatible_matrix(&matrices::SEPIA));
        assert!(!is_premul_compatible_matrix(&matrices::ALPHA_TO_BLACK));
    }

    #[test]
    fn premul_compatible_path_matches_straight_alpha_path() {
        let pixel = PremulRgba8 {
            r: 80,
            g: 32,
            b: 16,
            a: 128,
        };

        let premul = apply_premul_color_matrix_to_pixel(pixel, &matrices::SEPIA);
        let straight = apply_color_matrix_to_pixel(pixel, &matrices::SEPIA);

        assert_eq!(premul, straight);
    }

    #[test]
    fn premul_compatible_path_clamps_rgb_to_alpha() {
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

        let premul = apply_premul_color_matrix_to_pixel(pixel, &matrix);
        let straight = apply_color_matrix_to_pixel(pixel, &matrix);

        assert_eq!(premul, straight);
        assert_eq!(
            premul,
            PremulRgba8 {
                r: 128,
                g: 0,
                b: 0,
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

    #[test]
    fn execution_updates_transparency_flag_for_opaque_output() {
        let mut pixmap = Pixmap::from_parts_with_opacity(
            alloc::vec![PremulRgba8 {
                r: 128,
                g: 0,
                b: 0,
                a: 128,
            }],
            1,
            1,
            true,
        );
        let matrix = [
            1.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 1.0,
        ];

        apply_color_matrix(&mut pixmap, &matrix);

        assert!(!pixmap.may_have_transparency());
        assert_eq!(
            pixmap.sample(0, 0),
            PremulRgba8 {
                r: 255,
                g: 0,
                b: 0,
                a: 255,
            }
        );
    }
}
