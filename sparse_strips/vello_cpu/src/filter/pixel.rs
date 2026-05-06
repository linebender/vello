// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Pixel helpers for CPU filter effects.

use vello_common::peniko::color::PremulRgba8;
#[cfg(not(feature = "std"))]
use vello_common::peniko::kurbo::common::FloatFuncs as _;

const INV_255: f32 = 1.0 / 255.0;

/// Convert a u8 color component to normalized f32.
#[inline]
pub(super) fn u8_to_norm(value: u8) -> f32 {
    f32::from(value) * INV_255
}

/// Convert a normalized f32 color component to u8.
#[inline]
pub(super) fn norm_to_u8(value: f32) -> u8 {
    (clamp_unit(value) * 255.0).round() as u8
}

/// Premultiply a u8 color component by a u8 alpha value.
#[inline]
pub(super) fn premultiply_u8(channel: u8, alpha: u8) -> u8 {
    ((u16::from(channel) * u16::from(alpha)) / 255) as u8
}

/// Convert a premultiplied RGBA8 pixel to normalized straight-alpha components.
#[inline]
pub(super) fn premul_rgba8_to_straight_f32(pixel: PremulRgba8) -> [f32; 4] {
    let a = u8_to_norm(pixel.a);

    match pixel.a {
        0 => [0.0, 0.0, 0.0, 0.0],
        255 => [
            u8_to_norm(pixel.r),
            u8_to_norm(pixel.g),
            u8_to_norm(pixel.b),
            1.0,
        ],
        _ => {
            let inv_alpha = 1.0 / a;
            [
                u8_to_norm(pixel.r) * inv_alpha,
                u8_to_norm(pixel.g) * inv_alpha,
                u8_to_norm(pixel.b) * inv_alpha,
                a,
            ]
        }
    }
}

/// Convert normalized straight-alpha components to a premultiplied RGBA8 pixel.
#[inline]
pub(super) fn straight_f32_to_premul_rgba8(r: f32, g: f32, b: f32, a: f32) -> PremulRgba8 {
    let a = clamp_unit(a);

    PremulRgba8 {
        r: norm_to_u8(r * a),
        g: norm_to_u8(g * a),
        b: norm_to_u8(b * a),
        a: norm_to_u8(a),
    }
}

#[inline]
pub(super) fn clamp_unit(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u8_to_norm_converts_endpoints() {
        assert_eq!(u8_to_norm(0), 0.0);
        assert!((u8_to_norm(255) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn norm_to_u8_rounds_to_nearest() {
        assert_eq!(norm_to_u8(0.0), 0);
        assert_eq!(norm_to_u8(1.0), 255);
        assert_eq!(norm_to_u8(0.5), 128);
    }

    #[test]
    fn norm_to_u8_clamps() {
        assert_eq!(norm_to_u8(-1.0), 0);
        assert_eq!(norm_to_u8(2.0), 255);
    }

    #[test]
    fn premul_to_straight_handles_transparent_black() {
        assert_eq!(
            premul_rgba8_to_straight_f32(PremulRgba8::from_u32(0)),
            [0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn straight_to_premul_premultiplies_rgb() {
        assert_eq!(
            straight_f32_to_premul_rgba8(1.0, 0.5, 0.0, 0.5),
            PremulRgba8 {
                r: 128,
                g: 64,
                b: 0,
                a: 128,
            }
        );
    }
}
