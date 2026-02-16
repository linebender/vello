// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The gaussian blur filter.

use crate::filter_effects::EdgeMode;
use crate::kurbo::Affine;
use crate::util::extract_scales;
use core::f32::consts::E;
#[cfg(not(feature = "std"))]
use peniko::kurbo::common::FloatFuncs as _;

/// Scale a blur's standard deviation uniformly based on the transformation.
///
/// Extracts the scale factors from the transformation matrix using SVD and
/// averages them to get a uniform scale factor for the blur radius.
///
/// # Arguments
/// * `std_deviation` - The blur standard deviation in user space
/// * `transform` - The transformation matrix to extract scale from
///
/// # Returns
/// The scaled standard deviation in device space
pub(crate) fn transform_blur_params(std_deviation: f32, transform: &Affine) -> f32 {
    let (scale_x, scale_y) = extract_scales(transform);
    let uniform_scale = (scale_x + scale_y) / 2.0;
    // TODO: Support separate std_deviation for x and y axes (std_deviation_x, std_deviation_y)
    // to properly handle non-uniform scaling. This would eliminate the need for uniform_scale
    // and allow blur to scale independently along each axis.
    std_deviation * uniform_scale
}

/// Maximum size of the Gaussian kernel (must be odd and equal to or smaller than [`u8::MAX`]).
///
/// The multi-scale decimation algorithm guarantees that kernel size never exceeds this value.
/// Decimation stops when remaining variance ≤ 4.0 (σ ≤ 2.0), which produces kernels of size
/// at most 13 (radius = ceil(3σ) = 6, size = 1 + 2×6 = 13).
// Keep in sync with MAX_KERNEL_SIZE in vello_sparse_shaders/shaders/filters.wgsl
pub const MAX_KERNEL_SIZE: usize = 13;

#[cfg(test)]
const _: () = const {
    if MAX_KERNEL_SIZE.is_multiple_of(2) {
        panic!("`MAX_KERNEL_SIZE` must be odd");
    }
    if MAX_KERNEL_SIZE > u8::MAX as usize {
        panic!("`MAX_KERNEL_SIZE` must be less than or equal to `u8::MAX`");
    }
};

/// A gaussian blur.
#[derive(Debug)]
pub struct GaussianBlur {
    /// The standard deviation.
    pub std_deviation: f32,
    /// Number of 2× decimation levels to use (0 means no decimation, direct convolution).
    pub n_decimations: usize,
    /// Pre-computed Gaussian kernel weights for the reduced blur.
    /// Only the first `kernel_size` elements are valid.
    pub kernel: [f32; MAX_KERNEL_SIZE],
    /// Actual length of the kernel (rest is padding up to `MAX_KERNEL_SIZE`).
    pub kernel_size: u8,
    /// Edge mode for handling out-of-bounds sampling.
    pub edge_mode: EdgeMode,
}

impl GaussianBlur {
    /// Create a new Gaussian blur filter with the specified standard deviation.
    ///
    /// This precomputes the decimation plan, kernel, and radius for optimal performance.
    pub fn new(std_deviation: f32, edge_mode: EdgeMode) -> Self {
        let (n_decimations, kernel, kernel_size) = plan_decimated_blur(std_deviation);

        Self {
            std_deviation,
            edge_mode,
            n_decimations,
            kernel,
            kernel_size,
        }
    }
}

/// Compute the blur execution plan based on standard deviation.
///
/// Returns (`n_decimations`, `kernel`, `kernel_size`):
/// - `n_decimations`: Number of 2× downsampling steps to perform (per axis)
/// - `kernel`: Pre-computed Gaussian kernel weights (fixed-size array)
/// - `kernel_size`: Actual length of the kernel (rest is zero-padded)
pub fn plan_decimated_blur(std_deviation: f32) -> (usize, [f32; MAX_KERNEL_SIZE], u8) {
    if std_deviation <= 0.0 {
        // Invalid standard deviation, return identity kernel (no blur)
        let mut kernel = [0.0; MAX_KERNEL_SIZE];
        kernel[0] = 1.0;
        return (0, kernel, 1);
    }

    // Compute decimation plan using variance analysis.
    // Variance (σ²) has the additive property: applying two blurs sequentially
    // adds their variances together. We use this to decompose the blur.
    //
    // Mathematical Foundation: From probability theory, convolving two Gaussians
    // G(σ₁) ⊗ G(σ₂) = G(√(σ₁² + σ₂²)). This means variance is additive: σ²_total = σ²_1 + σ²_2.
    // Rearranging: σ²_2 = σ²_total - σ²_1, allowing us to decompose the target blur.
    let variance = std_deviation * std_deviation;
    let mut n_decimations = 0;
    let mut remaining_variance = variance;

    // Each decimation step:
    // 1. Applies [1,3,3,1] blur which adds variance = 0.75 per axis
    // 2. Downsamples by 2× per axis, adding variance = 0.75 × 4 = 3.0 total
    // 3. The 2× downsampling scales the remaining variance by 0.25 (= 1/2²)
    // Stop when remaining variance ≤ 4.0 (σ ≈ 2), as further decimation isn't beneficial.
    while remaining_variance > 4.0 {
        remaining_variance = (remaining_variance - 3.0) * 0.25;
        n_decimations += 1;
    }
    // Compute the reduced standard deviation to apply at the decimated resolution
    let remaining_sigma = remaining_variance.sqrt();
    // Compute Gaussian kernel for the reduced blur
    let (kernel, kernel_size) = compute_gaussian_kernel(remaining_sigma);

    (n_decimations, kernel, kernel_size)
}

/// Compute 1D Gaussian kernel weights for separable convolution.
///
/// Returns (`kernel_weights`, `kernel_size`) where `kernel_size = 2×radius + 1`.
/// The kernel is stored in a fixed-size array to avoid heap allocation.
/// Uses the standard Gaussian formula: G(x) = exp(-x² / (2σ²)), normalized to sum to 1.
pub fn compute_gaussian_kernel(std_deviation: f32) -> ([f32; MAX_KERNEL_SIZE], u8) {
    // Use radius = 3σ to capture 99.7% of the Gaussian distribution.
    // Beyond ±3σ, the Gaussian values are negligible (<0.3%).
    let radius = (3.0 * std_deviation).ceil() as usize;
    let kernel_size = (1 + radius * 2).min(MAX_KERNEL_SIZE) as u8;

    let mut kernel = [0.0; MAX_KERNEL_SIZE];
    // Compute Gaussian weights using the formula: G(x) = exp(-x² / (2σ²))
    // This creates a symmetric bell curve centered at the middle of the kernel.
    let gaussian_denominator = 2.0 * std_deviation * std_deviation;
    let mut sum = 0.0;
    let kernel_center = (kernel_size / 2) as f32;
    for (i, weight) in kernel.iter_mut().enumerate().take(usize::from(kernel_size)) {
        // Compute distance from center (0 at center, increases outward)
        let x = (i as f32) - kernel_center;
        // Apply Gaussian formula: weight decreases exponentially with squared distance
        *weight = E.powf(-x * x / gaussian_denominator);
        sum += *weight;
    }

    // Normalize weights to sum to 1.0, ensuring the blur doesn't change overall brightness.
    // Without normalization, blurring a uniform gray area could make it brighter/darker.
    let scale = 1.0 / sum;
    for weight in kernel.iter_mut().take(usize::from(kernel_size)) {
        *weight *= scale;
    }

    (kernel, kernel_size)
}

#[cfg(test)]
mod tests {
    use crate::filter::gaussian_blur::{compute_gaussian_kernel, plan_decimated_blur};

    /// Test Gaussian kernel computation for small σ.
    #[test]
    fn test_gaussian_kernel_small_sigma() {
        let (kernel, size) = compute_gaussian_kernel(1.0);
        // For σ=1.0, radius = ceil(3.0) = 3, size = 2*3+1 = 7
        assert_eq!(size, 7);

        // Kernel should be symmetric
        for i in 0..size / 2 {
            assert!((kernel[usize::from(i)] - kernel[usize::from(size - 1 - i)]).abs() < 1e-6);
        }

        // Kernel should sum to 1.0 (normalized)
        let sum: f32 = kernel.iter().take(usize::from(size)).sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Center should be the largest weight
        let center_idx = size / 2;
        for i in 0..size {
            if i != center_idx {
                assert!(kernel[usize::from(center_idx)] >= kernel[usize::from(i)]);
            }
        }
    }

    /// Test Gaussian kernel computation for very small σ (near-zero).
    #[test]
    fn test_gaussian_kernel_very_small_sigma() {
        let (kernel, size) = compute_gaussian_kernel(0.1);
        // For σ=0.1, radius = ceil(0.3) = 1, size = 3
        assert_eq!(size, 3);
        // Should sum to 1.0
        let sum: f32 = kernel.iter().take(usize::from(size)).sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Center weight should be dominant for very small σ
        assert!(kernel[1] > 0.9); // Center is highly weighted
    }

    /// Test Gaussian kernel for fractional σ.
    #[test]
    fn test_gaussian_kernel_fractional_sigma() {
        let (kernel, size) = compute_gaussian_kernel(0.5);
        // For σ=0.5, radius = ceil(1.5) = 2, size = 5
        assert_eq!(size, 5);

        // Should still sum to 1.0
        let sum: f32 = kernel.iter().take(usize::from(size)).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    /// Test decimation plan for small blur (no decimation).
    #[test]
    fn test_plan_no_decimation() {
        let (n_decimations, _kernel, _size) = plan_decimated_blur(1.0);
        // σ=1.0 → variance=1.0, should not decimate
        assert_eq!(n_decimations, 0);
    }

    /// Test decimation plan for medium blur (some decimation).
    #[test]
    fn test_plan_with_decimation() {
        let (n_decimations, _kernel, _size) = plan_decimated_blur(5.0);
        // σ=5.0 → variance=25.0, should decimate
        assert_eq!(n_decimations, 2);
    }

    /// Test decimation plan at boundary (σ=2.0).
    #[test]
    fn test_plan_decimation_boundary() {
        let (n_decimations, _kernel, _size) = plan_decimated_blur(2.0);
        // σ=2.0 → variance=4.0, right at the boundary
        assert_eq!(n_decimations, 0);
    }

    /// Test decimation plan for negative σ (invalid, should return identity).
    #[test]
    fn test_plan_negative_sigma() {
        let (n_decimations, kernel, size) = plan_decimated_blur(-1.0);
        assert_eq!(n_decimations, 0);
        assert_eq!(size, 1);
        assert!((kernel[0] - 1.0).abs() < 1e-6);
    }
}
