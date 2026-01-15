// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Drop shadow filter implementation.
//!
//! This implements the feDropShadow primitive from SVG Filter Effects 2.
//! The drop shadow effect is a shorthand for a commonly used sequence of filter operations:
//! 1. Extract alpha channel
//! 2. Offset the alpha
//! 3. Blur the offset alpha
//! 4. Composite shadow color with blurred alpha
//! 5. Composite shadow with original graphic
//!
//! @see <https://drafts.fxtf.org/filter-effects-2/#feDropShadowElement>

use super::FilterEffect;
use super::gaussian_blur::{MAX_KERNEL_SIZE, apply_blur, plan_decimated_blur};
use super::shift::offset_pixels;
use crate::layer_manager::LayerManager;
use vello_common::color::{AlphaColor, Srgb};
use vello_common::filter_effects::EdgeMode;
use vello_common::peniko::color::PremulRgba8;
#[cfg(not(feature = "std"))]
use vello_common::peniko::kurbo::common::FloatFuncs as _;
use vello_common::pixmap::Pixmap;

pub(crate) struct DropShadow {
    pub dx: f32,
    pub dy: f32,
    pub color: AlphaColor<Srgb>,
    /// Standard deviation for the blur (for reference/debugging).
    std_deviation: f32,
    /// Edge mode for blur sampling.
    edge_mode: EdgeMode,
    /// Number of 2x2 decimation levels to use (0 means direct convolution).
    n_decimations: usize,
    /// Pre-computed Gaussian kernel weights for the reduced blur.
    /// Only the first `kernel_size` elements are valid.
    kernel: [f32; MAX_KERNEL_SIZE],
    /// Actual length of the kernel (kernel is padded to `MAX_KERNEL_SIZE`).
    kernel_size: u8,
}

impl DropShadow {
    /// Create a new drop shadow filter with the specified parameters.
    ///
    /// This precomputes the blur decimation plan and kernel for optimal performance.
    pub(crate) fn new(
        dx: f32,
        dy: f32,
        std_deviation: f32,
        edge_mode: EdgeMode,
        color: AlphaColor<Srgb>,
    ) -> Self {
        // Precompute blur plan (same logic as GaussianBlur::new)
        let (n_decimations, kernel, kernel_size) = plan_decimated_blur(std_deviation);

        Self {
            dx,
            dy,
            color,
            std_deviation,
            edge_mode,
            n_decimations,
            kernel,
            kernel_size,
        }
    }
}

impl FilterEffect for DropShadow {
    fn execute_lowp(&self, pixmap: &mut Pixmap, layer_manager: &mut LayerManager) {
        apply_drop_shadow(
            pixmap,
            self.dx,
            self.dy,
            self.std_deviation,
            self.n_decimations,
            &self.kernel[..usize::from(self.kernel_size)],
            self.color,
            self.edge_mode,
            layer_manager,
        );
    }

    fn execute_highp(&self, pixmap: &mut Pixmap, layer_manager: &mut LayerManager) {
        // TODO: Currently only lowp is implemented and used for highp as well.
        // This needs to be updated to use proper high-precision arithmetic.
        Self::execute_lowp(self, pixmap, layer_manager);
    }
}

/// Apply drop shadow effect.
///
/// This is the main entry point that splits the drop shadow operation into well-defined steps:
/// 1. Offset the shadow pixels
/// 2. Blur the already-offset shadow
/// 3. Apply shadow color and composite with original
fn apply_drop_shadow(
    pixmap: &mut Pixmap,
    dx: f32,
    dy: f32,
    std_deviation: f32,
    n_decimations: usize,
    kernel: &[f32],
    color: AlphaColor<Srgb>,
    edge_mode: EdgeMode,
    layer_manager: &mut LayerManager,
) {
    // Clone pixmap to create shadow buffer
    let mut shadow_pixmap = pixmap.clone();

    // Step 1: Offset the shadow pixels
    offset_pixels(&mut shadow_pixmap, dx, dy);

    // Step 2: Blur the already-offset shadow
    if std_deviation > 0.0 {
        let scratch =
            layer_manager.get_scratch_buffer(shadow_pixmap.width(), shadow_pixmap.height());
        apply_blur(
            &mut shadow_pixmap,
            scratch,
            n_decimations,
            kernel,
            edge_mode,
        );
    }

    // Step 3: Apply shadow color and composite with original
    compose_shadow_direct(&shadow_pixmap, pixmap, color);
}

/// Apply shadow color and composite with original.
///
/// The shadow has already been offset and blurred, so this simply applies
/// the shadow color to the alpha channel and composites using source-over.
fn compose_shadow_direct(shadow: &Pixmap, dst: &mut Pixmap, color: AlphaColor<Srgb>) {
    let width = dst.width();
    let height = dst.height();

    // Precompute shadow color components
    let shadow_r = (color.components[0] * 255.0).round() as u8;
    let shadow_g = (color.components[1] * 255.0).round() as u8;
    let shadow_b = (color.components[2] * 255.0).round() as u8;

    for y in 0..height {
        for x in 0..width {
            // Sample alpha directly (shadow is already offset)
            let alpha = shadow.sample(x, y).a;

            // Apply shadow color to alpha
            let shadow_alpha = (u8_to_norm(alpha) * color.components[3]).min(1.0);
            let final_alpha = norm_to_u8(shadow_alpha);

            // Premultiply RGB by alpha as required by PremulRgba8
            let alpha_u16 = u16::from(final_alpha);
            let premultiply = |channel: u8| ((u16::from(channel) * alpha_u16) / 255) as u8;

            let colored_shadow = PremulRgba8 {
                r: premultiply(shadow_r),
                g: premultiply(shadow_g),
                b: premultiply(shadow_b),
                a: final_alpha,
            };

            // Read original and composite: original over shadow
            let original_pixel = dst.sample(x, y);
            let result = compose_src_over(original_pixel, colored_shadow);

            dst.set_pixel(x, y, result);
        }
    }
}

/// Composite two pixels using Porter-Duff "source over" operator.
///
/// Composes the source pixel over the destination pixel using premultiplied
/// alpha blending. Returns the composited result.
///
/// Formula for premultiplied colors: `result = src + dst * (1 - src_alpha)`
fn compose_src_over(src: PremulRgba8, dst: PremulRgba8) -> PremulRgba8 {
    let src_a = u8_to_norm(src.a);

    PremulRgba8 {
        r: src_over_channel(src.r, dst.r, src_a),
        g: src_over_channel(src.g, dst.g, src_a),
        b: src_over_channel(src.b, dst.b, src_a),
        a: src_over_channel(src.a, dst.a, src_a),
    }
}

/// Blend a single channel using Porter-Duff "source over" operator.
///
/// For premultiplied colors, the formula is: `result = src + dst * (1 - src_alpha)`
#[inline]
fn src_over_channel(src: u8, dst: u8, src_alpha: f32) -> u8 {
    let result = u8_to_norm(src) + u8_to_norm(dst) * (1.0 - src_alpha);
    norm_to_u8(result)
}

/// Convert a u8 color component (0-255) to normalized f32 (0.0-1.0).
#[inline]
fn u8_to_norm(value: u8) -> f32 {
    value as f32 / 255.0
}

/// Convert a normalized f32 (0.0-1.0) to u8 color component (0-255).
#[inline]
fn norm_to_u8(value: f32) -> u8 {
    (value * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use vello_common::color::Srgb;

    /// Test `u8_to_norm` conversion.
    #[test]
    fn test_u8_to_norm() {
        assert_eq!(u8_to_norm(0), 0.0);
        assert!((u8_to_norm(255) - 1.0).abs() < 1e-6);
    }

    /// Test `norm_to_u8` conversion.
    #[test]
    fn test_norm_to_u8() {
        assert_eq!(norm_to_u8(0.0), 0);
        assert_eq!(norm_to_u8(1.0), 255);
        assert_eq!(norm_to_u8(0.5), 128); // 0.5 * 255 = 127.5 → 128
    }

    /// Test round-trip conversion u8 → norm → u8.
    #[test]
    fn test_conversion_roundtrip() {
        for value in [0, 1, 50, 127, 128, 200, 254, 255] {
            let normalized = u8_to_norm(value);
            let back = norm_to_u8(normalized);
            assert_eq!(back, value);
        }
    }

    /// Test Porter-Duff source-over with fully opaque source.
    #[test]
    fn test_compose_src_over_opaque_source() {
        let src = PremulRgba8 {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        }; // Opaque red
        let dst = PremulRgba8 {
            r: 0,
            g: 255,
            b: 0,
            a: 255,
        }; // Opaque green

        let result = compose_src_over(src, dst);
        // Opaque source should completely cover destination
        assert_eq!(result.r, 255);
        assert_eq!(result.g, 0);
        assert_eq!(result.b, 0);
        assert_eq!(result.a, 255);
    }

    /// Test Porter-Duff source-over with transparent source.
    #[test]
    fn test_compose_src_over_transparent_source() {
        let src = PremulRgba8 {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        };
        let dst = PremulRgba8 {
            r: 0,
            g: 255,
            b: 0,
            a: 255,
        };

        let result = compose_src_over(src, dst);
        // Transparent source should leave destination unchanged
        assert_eq!(result.r, 0);
        assert_eq!(result.g, 255);
        assert_eq!(result.b, 0);
        assert_eq!(result.a, 255);
    }

    /// Test Porter-Duff source-over with semi-transparent source.
    #[test]
    fn test_compose_src_over_semi_transparent() {
        let src = PremulRgba8 {
            r: 128,
            g: 0,
            b: 0,
            a: 128,
        }; // 50% red (premul)
        let dst = PremulRgba8 {
            r: 0,
            g: 128,
            b: 0,
            a: 128,
        }; // 50% green (premul)

        let result = compose_src_over(src, dst);
        // Result should blend src + dst*(1-src_alpha)
        // r: 128 + 0*(1-0.5) = 128
        // g: 0 + 128*0.5 = 64
        // a: 128 + 128*0.5 = 192
        assert_eq!(
            result,
            PremulRgba8 {
                r: 128,
                g: 64,
                b: 0,
                a: 192,
            }
        );
    }

    /// Test `compose_shadow_direct` applies color correctly.
    #[test]
    fn test_compose_shadow_color() {
        let mut shadow_pixmap = Pixmap::new(2, 2);
        let mut dst_pixmap = Pixmap::new(2, 2);

        // Shadow has alpha=255 at (0,0)
        shadow_pixmap.set_pixel(
            0,
            0,
            PremulRgba8 {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            },
        );

        let shadow_color = AlphaColor {
            components: [1.0, 0.0, 0.0, 1.0], // Red
            cs: std::marker::PhantomData::<Srgb>,
        };

        compose_shadow_direct(&shadow_pixmap, &mut dst_pixmap, shadow_color);

        // Shadow at (0,0) should be red
        let result = dst_pixmap.sample(0, 0);
        assert_eq!(result.r, 255);
        assert_eq!(result.g, 0);
        assert_eq!(result.b, 0);
        assert_eq!(result.a, 255);
    }
}
