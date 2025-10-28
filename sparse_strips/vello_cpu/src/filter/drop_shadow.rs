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

use super::{FilterBuffer, FilterEffect};
use crate::fine::Numeric;
use alloc::vec::Vec;
use vello_common::color::{AlphaColor, Srgb};

pub(crate) struct DropShadow {
    pub dx: f32,
    pub dy: f32,
    pub std_deviation: f32,
    pub color: AlphaColor<Srgb>,
}

impl DropShadow {
    /// Create a new drop shadow filter with the specified parameters.
    pub(crate) fn new(dx: f32, dy: f32, std_deviation: f32, color: AlphaColor<Srgb>) -> Self {
        Self {
            dx,
            dy,
            std_deviation,
            color,
        }
    }
}

/// Compute 1D Gaussian kernel weights for separable convolution.
fn compute_gaussian_kernel(std_deviation: f32) -> (Vec<f32>, usize) {
    let radius = (std_deviation * 3.0).ceil() as usize;
    let radius = radius.max(1);

    let kernel_size = radius * 2 + 1;
    let mut weights = alloc::vec![0.0; kernel_size];

    let sigma_sq_2 = 2.0 * std_deviation * std_deviation;
    let mut sum = 0.0;

    for i in 0..kernel_size {
        let x = (i as f32) - (radius as f32);
        weights[i] = (-x * x / sigma_sq_2).exp();
        sum += weights[i];
    }

    for weight in &mut weights {
        *weight /= sum;
    }

    (weights, radius)
}

/// Apply horizontal blur pass.
fn blur_horizontal<T: Numeric>(
    src: &FilterBuffer<T>,
    dst: &mut FilterBuffer<T>,
    kernel: &[f32],
    radius: usize,
) {
    let width = src.width();
    let height = src.height();

    for y in 0..height {
        for x in 0..width {
            let mut rgba = [0.0_f32; 4];

            for (i, &weight) in kernel.iter().enumerate() {
                let offset = i as i32 - radius as i32;
                let sample_x = (x as i32 + offset).clamp(0, width as i32 - 1) as u16;
                let pixel = src.get_pixel(sample_x, y);

                for c in 0..4 {
                    rgba[c] += pixel[c].to_f32() * weight;
                }
            }

            dst.set_pixel(
                x,
                y,
                [
                    T::from_f32(rgba[0]),
                    T::from_f32(rgba[1]),
                    T::from_f32(rgba[2]),
                    T::from_f32(rgba[3]),
                ],
            );
        }
    }
}

/// Apply vertical blur pass.
fn blur_vertical<T: Numeric>(
    src: &FilterBuffer<T>,
    dst: &mut FilterBuffer<T>,
    kernel: &[f32],
    radius: usize,
) {
    let width = src.width();
    let height = src.height();

    for y in 0..height {
        for x in 0..width {
            let mut rgba = [0.0_f32; 4];

            for (i, &weight) in kernel.iter().enumerate() {
                let offset = i as i32 - radius as i32;
                let sample_y = (y as i32 + offset).clamp(0, height as i32 - 1) as u16;
                let pixel = src.get_pixel(x, sample_y);

                for c in 0..4 {
                    rgba[c] += pixel[c].to_f32() * weight;
                }
            }

            dst.set_pixel(
                x,
                y,
                [
                    T::from_f32(rgba[0]),
                    T::from_f32(rgba[1]),
                    T::from_f32(rgba[2]),
                    T::from_f32(rgba[3]),
                ],
            );
        }
    }
}

impl FilterEffect for DropShadow {
    fn apply_u8(&self, buffer: &mut FilterBuffer<u8>) {
        let width = buffer.width();
        let height = buffer.height();

        // Step 1: Extract alpha channel and store original
        let mut shadow_buffer = FilterBuffer::new(width, height);
        let mut original_buffer = FilterBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let pixel = buffer.get_pixel(x, y);
                let alpha = pixel[3];

                // Store original
                original_buffer.set_pixel(x, y, pixel);

                // Create shadow from alpha (black with original alpha)
                shadow_buffer.set_pixel(x, y, [0, 0, 0, alpha]);
            }
        }

        // Step 2: Offset the shadow
        let mut offset_buffer = FilterBuffer::new(width, height);
        let dx_pixels = self.dx.round() as i32;
        let dy_pixels = self.dy.round() as i32;

        for y in 0..height {
            for x in 0..width {
                let src_x = x as i32 - dx_pixels;
                let src_y = y as i32 - dy_pixels;

                if src_x >= 0 && src_x < width as i32 && src_y >= 0 && src_y < height as i32 {
                    let pixel = shadow_buffer.get_pixel(src_x as u16, src_y as u16);
                    offset_buffer.set_pixel(x, y, pixel);
                } else {
                    // Outside bounds = transparent
                    offset_buffer.set_pixel(x, y, [0, 0, 0, 0]);
                }
            }
        }

        // Step 3: Blur the shadow (if std_deviation > 0)
        if self.std_deviation > 0.0 {
            let (kernel, radius) = compute_gaussian_kernel(self.std_deviation);
            let mut temp_buffer = FilterBuffer::new(width, height);

            blur_horizontal(&offset_buffer, &mut temp_buffer, &kernel, radius);
            blur_vertical(&temp_buffer, &mut offset_buffer, &kernel, radius);
        }

        // Step 4: Apply shadow color
        let shadow_r = (self.color.components[0] * 255.0).round() as u8;
        let shadow_g = (self.color.components[1] * 255.0).round() as u8;
        let shadow_b = (self.color.components[2] * 255.0).round() as u8;

        for y in 0..height {
            for x in 0..width {
                let pixel = offset_buffer.get_pixel(x, y);
                let alpha = pixel[3];

                // Apply shadow color with the blurred alpha
                let shadow_alpha = ((alpha as f32 / 255.0) * self.color.components[3]).min(1.0);
                let final_alpha = (shadow_alpha * 255.0).round() as u8;

                offset_buffer.set_pixel(x, y, [shadow_r, shadow_g, shadow_b, final_alpha]);
            }
        }

        // Step 5: Composite shadow with original (shadow under original)
        for y in 0..height {
            for x in 0..width {
                let shadow = offset_buffer.get_pixel(x, y);
                let original = original_buffer.get_pixel(x, y);

                // Porter-Duff "over" operator: result = src over dst
                // We want: original over shadow
                let src_a = original[3] as f32 / 255.0;
                let dst_a = shadow[3] as f32 / 255.0;

                let out_a = src_a + dst_a * (1.0 - src_a);

                let (out_r, out_g, out_b) = if out_a > 0.0 {
                    let r = ((original[0] as f32 / 255.0) * src_a
                        + (shadow[0] as f32 / 255.0) * dst_a * (1.0 - src_a))
                        / out_a;
                    let g = ((original[1] as f32 / 255.0) * src_a
                        + (shadow[1] as f32 / 255.0) * dst_a * (1.0 - src_a))
                        / out_a;
                    let b = ((original[2] as f32 / 255.0) * src_a
                        + (shadow[2] as f32 / 255.0) * dst_a * (1.0 - src_a))
                        / out_a;
                    (
                        (r * 255.0).round() as u8,
                        (g * 255.0).round() as u8,
                        (b * 255.0).round() as u8,
                    )
                } else {
                    (0, 0, 0)
                };

                buffer.set_pixel(x, y, [out_r, out_g, out_b, (out_a * 255.0).round() as u8]);
            }
        }
    }

    fn apply_f32(&self, buffer: &mut FilterBuffer<f32>) {
        let width = buffer.width();
        let height = buffer.height();

        // Step 1: Extract alpha channel and store original
        let mut shadow_buffer = FilterBuffer::new(width, height);
        let mut original_buffer = FilterBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let pixel = buffer.get_pixel(x, y);
                let alpha = pixel[3];

                // Store original
                original_buffer.set_pixel(x, y, pixel);

                // Create shadow from alpha (black with original alpha)
                shadow_buffer.set_pixel(x, y, [0.0, 0.0, 0.0, alpha]);
            }
        }

        // Step 2: Offset the shadow
        let mut offset_buffer = FilterBuffer::new(width, height);
        let dx_pixels = self.dx.round() as i32;
        let dy_pixels = self.dy.round() as i32;

        for y in 0..height {
            for x in 0..width {
                let src_x = x as i32 - dx_pixels;
                let src_y = y as i32 - dy_pixels;

                if src_x >= 0 && src_x < width as i32 && src_y >= 0 && src_y < height as i32 {
                    let pixel = shadow_buffer.get_pixel(src_x as u16, src_y as u16);
                    offset_buffer.set_pixel(x, y, pixel);
                } else {
                    // Outside bounds = transparent
                    offset_buffer.set_pixel(x, y, [0.0, 0.0, 0.0, 0.0]);
                }
            }
        }

        // Step 3: Blur the shadow (if std_deviation > 0)
        if self.std_deviation > 0.0 {
            let (kernel, radius) = compute_gaussian_kernel(self.std_deviation);
            let mut temp_buffer = FilterBuffer::new(width, height);

            blur_horizontal(&offset_buffer, &mut temp_buffer, &kernel, radius);
            blur_vertical(&temp_buffer, &mut offset_buffer, &kernel, radius);
        }

        // Step 4: Apply shadow color
        let shadow_r = self.color.components[0];
        let shadow_g = self.color.components[1];
        let shadow_b = self.color.components[2];

        for y in 0..height {
            for x in 0..width {
                let pixel = offset_buffer.get_pixel(x, y);
                let alpha = pixel[3];

                // Apply shadow color with the blurred alpha
                let shadow_alpha = (alpha * self.color.components[3]).min(1.0);

                offset_buffer.set_pixel(x, y, [shadow_r, shadow_g, shadow_b, shadow_alpha]);
            }
        }

        // Step 5: Composite shadow with original (shadow under original)
        for y in 0..height {
            for x in 0..width {
                let shadow = offset_buffer.get_pixel(x, y);
                let original = original_buffer.get_pixel(x, y);

                // Porter-Duff "over" operator: result = src over dst
                // We want: original over shadow
                let src_a = original[3];
                let dst_a = shadow[3];

                let out_a = src_a + dst_a * (1.0 - src_a);

                let (out_r, out_g, out_b) = if out_a > 0.0 {
                    let r = (original[0] * src_a + shadow[0] * dst_a * (1.0 - src_a)) / out_a;
                    let g = (original[1] * src_a + shadow[1] * dst_a * (1.0 - src_a)) / out_a;
                    let b = (original[2] * src_a + shadow[2] * dst_a * (1.0 - src_a)) / out_a;
                    (r, g, b)
                } else {
                    (0.0, 0.0, 0.0)
                };

                buffer.set_pixel(x, y, [out_r, out_g, out_b, out_a]);
            }
        }
    }
}
