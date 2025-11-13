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
    /// Number of 2x2 decimation levels to use (0 means direct convolution).
    n_decimations: usize,
    /// Pre-computed Gaussian kernel weights for the reduced blur.
    /// Only the first `kernel_size` elements are valid.
    kernel: [f32; MAX_KERNEL_SIZE],
    /// Actual length of the kernel (kernel is padded to `MAX_KERNEL_SIZE`).
    kernel_size: usize,
}

impl DropShadow {
    /// Create a new drop shadow filter with the specified parameters.
    ///
    /// This precomputes the blur decimation plan and kernel for optimal performance.
    pub(crate) fn new(dx: f32, dy: f32, std_deviation: f32, color: AlphaColor<Srgb>) -> Self {
        // Precompute blur plan (same logic as GaussianBlur::new)
        let (n_decimations, kernel, kernel_size) = plan_decimated_blur(std_deviation);

        Self {
            dx,
            dy,
            color,
            std_deviation,
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
            &self.kernel[..self.kernel_size],
            self.color,
            layer_manager,
        );
    }

    fn execute_highp(&self, pixmap: &mut Pixmap, layer_manager: &mut LayerManager) {
        // TODO: Currently only lowp is implemented and used for highp as well.
        // This needs to be updated to use proper high-precision arithmetic.

        apply_drop_shadow(
            pixmap,
            self.dx,
            self.dy,
            self.std_deviation,
            self.n_decimations,
            &self.kernel[..self.kernel_size],
            self.color,
            layer_manager,
        );
    }
}

/// Apply drop shadow effect.
///
/// This is the main entry point that splits the drop shadow operation into
/// well-defined steps following the pattern from `gaussian_blur.rs`.
fn apply_drop_shadow(
    pixmap: &mut Pixmap,
    dx: f32,
    dy: f32,
    std_deviation: f32,
    n_decimations: usize,
    kernel: &[f32],
    color: AlphaColor<Srgb>,
    layer_manager: &mut LayerManager,
) {
    // Clone pixmap to create shadow buffer
    let mut shadow_pixmap = pixmap.clone();

    // Blur the shadow (without offsetting yet)
    if std_deviation > 0.0 {
        let scratch =
            layer_manager.get_scratch_buffer(shadow_pixmap.width(), shadow_pixmap.height());
        apply_blur(
            &mut shadow_pixmap,
            scratch,
            n_decimations,
            kernel,
            EdgeMode::None,
        );
    }

    // Apply offset, shadow color, and composite with original (fused loop)
    compose_shadow(&shadow_pixmap, pixmap, color, dx, dy);
}

/// Apply shadow color, offset, and composite with original in a single pass.
///
/// This fuses three operations for better performance:
/// 1. Applies spatial offset to the shadow (via coordinate transformation during sampling)
/// 2. Applies shadow color to the blurred alpha channel
/// 3. Composites the colored shadow with the original image
///
/// This loop fusion eliminates two full iterations and avoids creating intermediate
/// buffers for offset and colored shadow pixels.
fn compose_shadow(src: &Pixmap, dst: &mut Pixmap, color: AlphaColor<Srgb>, dx: f32, dy: f32) {
    let width = dst.width();
    let height = dst.height();

    // Precompute offset in pixels and shadow color components
    let dx_pixels = dx.round() as i32;
    let dy_pixels = dy.round() as i32;
    let shadow_r = (color.components[0] * 255.0).round() as u8;
    let shadow_g = (color.components[1] * 255.0).round() as u8;
    let shadow_b = (color.components[2] * 255.0).round() as u8;

    for y in 0..height {
        for x in 0..width {
            // Sample alpha with offset and bounds checking
            let alpha = sample_alpha_with_offset(src, x, y, dx_pixels, dy_pixels);

            // Apply shadow color on-the-fly
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

            // Read original and composite immediately
            let original_pixel = dst.sample(x, y);
            let result = compose_src_over(original_pixel, colored_shadow);

            dst.set_pixel(x, y, result);
        }
    }
}

/// Sample alpha channel with spatial offset and bounds checking.
///
/// Returns 0 (transparent) if the offset position is out of bounds.
/// This is useful for filter effects that need to sample from offset positions.
#[inline]
fn sample_alpha_with_offset(pixmap: &Pixmap, x: u16, y: u16, dx: i32, dy: i32) -> u8 {
    let shadow_x = x as i32 - dx;
    let shadow_y = y as i32 - dy;
    let width = pixmap.width();
    let height = pixmap.height();

    if shadow_x >= 0 && shadow_x < width as i32 && shadow_y >= 0 && shadow_y < height as i32 {
        pixmap.sample(shadow_x as u16, shadow_y as u16).a
    } else {
        0
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
