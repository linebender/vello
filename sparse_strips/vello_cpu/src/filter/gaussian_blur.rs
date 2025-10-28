// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Gaussian blur filter implementation using separable convolution.

use super::{FilterBuffer, FilterEffect};
use crate::fine::Numeric;
use alloc::vec::Vec;

pub(crate) struct GaussianBlur {
    pub std_deviation: f32,
}

impl GaussianBlur {
    /// Create a new Gaussian blur filter with the specified standard deviation.
    pub(crate) fn new(std_deviation: f32) -> Self {
        Self { std_deviation }
    }
}

/// Compute 1D Gaussian kernel weights for separable convolution.
///
/// Returns (kernel_weights, radius) where radius is the kernel half-width.
fn compute_gaussian_kernel(std_deviation: f32) -> (Vec<f32>, usize) {
    // Kernel radius is typically 3*sigma (covers 99.7% of distribution)
    let radius = (std_deviation * 3.0).ceil() as usize;
    let radius = radius.max(1); // At least radius of 1

    let kernel_size = radius * 2 + 1;
    let mut weights = alloc::vec![0.0; kernel_size];

    // Compute Gaussian weights: exp(-(x^2) / (2*sigma^2))
    let sigma_sq_2 = 2.0 * std_deviation * std_deviation;
    let mut sum = 0.0;

    for i in 0..kernel_size {
        let x = (i as f32) - (radius as f32);
        weights[i] = (-x * x / sigma_sq_2).exp();
        sum += weights[i];
    }

    // Normalize weights so they sum to 1.0
    for weight in &mut weights {
        *weight /= sum;
    }

    (weights, radius)
}

/// Apply horizontal blur pass (separable convolution).
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
            let mut rgba = [0.0f32; 4];

            // Convolve with kernel
            for (i, &weight) in kernel.iter().enumerate() {
                let offset = i as i32 - radius as i32;
                let sample_x = (x as i32 + offset).clamp(0, width as i32 - 1) as u16;
                let pixel = src.get_pixel(sample_x, y);

                for c in 0..4 {
                    rgba[c] += pixel[c].to_f32() * weight;
                }
            }

            // Convert back to T
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

/// Apply vertical blur pass (separable convolution).
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
            let mut rgba = [0.0f32; 4];

            // Convolve with kernel
            for (i, &weight) in kernel.iter().enumerate() {
                let offset = i as i32 - radius as i32;
                let sample_y = (y as i32 + offset).clamp(0, height as i32 - 1) as u16;
                let pixel = src.get_pixel(x, sample_y);

                for c in 0..4 {
                    rgba[c] += pixel[c].to_f32() * weight;
                }
            }

            // Convert back to T
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

impl FilterEffect for GaussianBlur {
    fn apply_u8(&self, buffer: &mut FilterBuffer<u8>) {
        // No blur if std_deviation is zero or negative
        if self.std_deviation <= 0.0 {
            return;
        }

        let (kernel, radius) = compute_gaussian_kernel(self.std_deviation);

        // Allocate temporary buffer for horizontal pass
        let mut temp = FilterBuffer::new(buffer.width(), buffer.height());

        // Two-pass separable convolution
        blur_horizontal(buffer, &mut temp, &kernel, radius);
        blur_vertical(&temp, buffer, &kernel, radius);
    }

    fn apply_f32(&self, buffer: &mut FilterBuffer<f32>) {
        // No blur if std_deviation is zero or negative
        if self.std_deviation <= 0.0 {
            return;
        }

        let (kernel, radius) = compute_gaussian_kernel(self.std_deviation);

        // Allocate temporary buffer for horizontal pass
        let mut temp = FilterBuffer::new(buffer.width(), buffer.height());

        // Two-pass separable convolution
        blur_horizontal(buffer, &mut temp, &kernel, radius);
        blur_vertical(&temp, buffer, &kernel, radius);
    }
}
