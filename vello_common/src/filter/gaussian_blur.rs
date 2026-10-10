// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The gaussian blur filter.

use alloc::vec::Vec;

use crate::filter_effects::EdgeMode;
use crate::kurbo::Affine;
use crate::util::extract_scales;
use core::f32::consts::E;
#[cfg(not(feature = "std"))]
use peniko::kurbo::common::FloatFuncs as _;

/// Scale a blur's standard deviations into device space.
///
/// A uniform blur (equal standard deviations) extracts the scale factors from the
/// transformation matrix using SVD and averages them into one uniform scale factor,
/// so rotating a uniform blur never changes it.
///
/// An anisotropic blur is the Gaussian with covariance `diag(σx², σy²)`; under the
/// linear part `A` of the transform it becomes the Gaussian with covariance
/// `A·diag(σx², σy²)·Aᵀ`. The diagonal of that matrix gives the device-space variances
/// per axis, so a 90° rotation swaps the two deviations and a non-uniform scale scales
/// each one on its own. The off-diagonal term (the tilt of a blur rotated by an angle
/// that is not a multiple of 90°) cannot be represented by an axis-aligned kernel and is
/// dropped.
///
/// # Arguments
/// * `std_deviation_x` - The blur standard deviation along the x-axis in user space
/// * `std_deviation_y` - The blur standard deviation along the y-axis in user space
/// * `transform` - The transformation matrix to extract scale from
///
/// # Returns
/// The scaled standard deviations in device space, as `(x, y)`.
pub(crate) fn transform_blur_params(
    std_deviation_x: f32,
    std_deviation_y: f32,
    transform: &Affine,
) -> (f32, f32) {
    if std_deviation_x == std_deviation_y {
        let (scale_x, scale_y) = extract_scales(transform);
        let uniform_scale = (scale_x + scale_y) / 2.0;
        let scaled = std_deviation_x * uniform_scale;
        return (scaled, scaled);
    }

    let [a, b, c, d, _, _] = transform.as_coeffs();
    let (a, b, c, d) = (a as f32, b as f32, c as f32, d as f32);
    let variance_x = std_deviation_x * std_deviation_x;
    let variance_y = std_deviation_y * std_deviation_y;

    (
        (a * a * variance_x + c * c * variance_y).sqrt(),
        (b * b * variance_x + d * d * variance_y).sqrt(),
    )
}

/// The axes a decimation level halves.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecimationAxes {
    /// Both axes are halved.
    Both,
    /// Only the x-axis is halved.
    X,
    /// Only the y-axis is halved.
    Y,
}

impl DecimationAxes {
    /// Whether the level halves the x-axis.
    #[inline]
    pub fn x(self) -> bool {
        matches!(self, Self::Both | Self::X)
    }

    /// Whether the level halves the y-axis.
    #[inline]
    pub fn y(self) -> bool {
        matches!(self, Self::Both | Self::Y)
    }
}

/// The decimation levels of a blur with `n_decimations_x` levels along the x-axis and
/// `n_decimations_y` along the y-axis, from the finest to the coarsest.
///
/// Both axes are halved while both still have levels left; after that only the axis
/// with the larger blur keeps being halved.
pub fn decimation_levels(
    n_decimations_x: usize,
    n_decimations_y: usize,
) -> impl DoubleEndedIterator<Item = DecimationAxes> + ExactSizeIterator + Clone {
    (0..n_decimations_x.max(n_decimations_y)).map(move |level| {
        match (level < n_decimations_x, level < n_decimations_y) {
            (true, true) => DecimationAxes::Both,
            (true, false) => DecimationAxes::X,
            (false, true) => DecimationAxes::Y,
            (false, false) => unreachable!("levels never exceed the larger count"),
        }
    })
}

/// Maximum size of the Gaussian kernel (must be odd and equal to or smaller than [`u8::MAX`]).
///
/// The multi-scale decimation algorithm guarantees that kernel size never exceeds this value.
/// Decimation stops when remaining variance ≤ 4.0 (σ ≤ 2.0), which produces kernels of size
/// at most 13 (radius = ceil(3σ) = 6, size = 1 + 2×6 = 13).
// Keep in sync with MAX_KERNEL_SIZE in vello_gpu_shaders/shaders/filter.wesl
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
///
/// Each axis has its own decimation plan and kernel, so the blur may differ per axis.
#[derive(Debug)]
pub struct GaussianBlur {
    /// The standard deviation along the x-axis.
    pub std_deviation_x: f32,
    /// The standard deviation along the y-axis.
    pub std_deviation_y: f32,
    /// Number of 2× decimation levels along the x-axis (0 means no decimation, direct
    /// convolution).
    pub n_decimations_x: usize,
    /// Number of 2× decimation levels along the y-axis.
    pub n_decimations_y: usize,
    /// Pre-computed Gaussian kernel weights for the reduced blur along the x-axis.
    /// Only the first `kernel_size_x` elements are valid.
    pub kernel_x: [f32; MAX_KERNEL_SIZE],
    /// Actual length of `kernel_x` (rest is padding up to `MAX_KERNEL_SIZE`).
    pub kernel_size_x: u8,
    /// Pre-computed Gaussian kernel weights for the reduced blur along the y-axis.
    /// Only the first `kernel_size_y` elements are valid.
    pub kernel_y: [f32; MAX_KERNEL_SIZE],
    /// Actual length of `kernel_y` (rest is padding up to `MAX_KERNEL_SIZE`).
    pub kernel_size_y: u8,
    /// Edge mode for handling out-of-bounds sampling.
    pub edge_mode: EdgeMode,
}

impl GaussianBlur {
    /// Create a new Gaussian blur filter with the specified standard deviations.
    ///
    /// This precomputes the decimation plan, kernel, and radius of each axis for optimal
    /// performance.
    pub fn new(std_deviation_x: f32, std_deviation_y: f32, edge_mode: EdgeMode) -> Self {
        let (n_decimations_x, kernel_x, kernel_size_x) = plan_decimated_blur(std_deviation_x);
        let (n_decimations_y, kernel_y, kernel_size_y) = plan_decimated_blur(std_deviation_y);

        Self {
            std_deviation_x,
            std_deviation_y,
            edge_mode,
            n_decimations_x,
            n_decimations_y,
            kernel_x,
            kernel_size_x,
            kernel_y,
            kernel_size_y,
        }
    }

    /// The decimation levels of this blur, from the finest to the coarsest.
    pub fn decimation_levels(
        &self,
    ) -> impl DoubleEndedIterator<Item = DecimationAxes> + ExactSizeIterator + Clone {
        decimation_levels(self.n_decimations_x, self.n_decimations_y)
    }

    /// The valid part of the x-axis kernel.
    #[inline]
    pub fn kernel_x(&self) -> &[f32] {
        &self.kernel_x[..usize::from(self.kernel_size_x)]
    }

    /// The valid part of the y-axis kernel.
    #[inline]
    pub fn kernel_y(&self) -> &[f32] {
        &self.kernel_y[..usize::from(self.kernel_size_y)]
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

    // Each decimation level blurs the image *twice* over the full round trip, and both passes
    // must be subtracted from the budget so the final result matches the target σ:
    // 1. The downscale applies a [1,3,3,1]/8 binomial filter (variance 0.75 in the current grid).
    // 2. The matching upscale reconstruction adds 0.75 variance too: each output samples
    //    neighbouring decimated pixels 0.5 and 1.5 original-grid pixels away from its centre,
    //    so 0.75*(0.5²) + 0.25*(1.5²) = 0.75.
    // So a level removes 0.75 + 0.75 = 1.5 of variance (in current-grid units) before the 2×
    // downsampling rescales the remaining variance by 0.25 (= 1/2²) into the next grid.
    while remaining_variance > 4.0 {
        remaining_variance = (remaining_variance - 1.5) * 0.25;
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

/// Tracks dimensions through a chain of downscale/upscale operations.
///
/// Each axis keeps its own history, so levels that halve one axis only can be undone in
/// reverse order like any other.
#[derive(Debug, Default)]
pub struct DecimationSizer {
    width: u16,
    height: u16,
    width_stack: Vec<u16>,
    height_stack: Vec<u16>,
}

impl DecimationSizer {
    /// Create a new sizer with the given initial dimensions.
    #[inline]
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            width,
            height,
            width_stack: Vec::new(),
            height_stack: Vec::new(),
        }
    }

    /// Reset the sizer so it can be reused.
    #[inline]
    pub fn reset(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
        self.width_stack.clear();
        self.height_stack.clear();
    }

    /// Returns the current logical dimensions.
    #[inline]
    pub fn current(&self) -> (u16, u16) {
        (self.width, self.height)
    }

    /// Apply a new downscale operation on both axes.
    #[inline]
    pub fn downscale(&mut self) -> (u16, u16) {
        self.downscale_axes(DecimationAxes::Both)
    }

    /// Apply a new upscale operation on both axes.
    #[inline]
    pub fn upscale(&mut self) -> (u16, u16) {
        self.upscale_axes(DecimationAxes::Both)
    }

    /// Apply a new downscale operation on the given axes.
    #[inline]
    pub fn downscale_axes(&mut self, axes: DecimationAxes) -> (u16, u16) {
        if axes.x() {
            self.width_stack.push(self.width);
            self.width = self.width.div_ceil(2);
        }
        if axes.y() {
            self.height_stack.push(self.height);
            self.height = self.height.div_ceil(2);
        }
        (self.width, self.height)
    }

    /// Apply a new upscale operation on the given axes, undoing the matching downscale.
    #[inline]
    pub fn upscale_axes(&mut self, axes: DecimationAxes) -> (u16, u16) {
        // Clamp because upscale can exceed target on odd dimensions (e.g., 5→3→6 > 5)
        if axes.x() {
            let target_w = self.width_stack.pop().unwrap();
            self.width = (self.width * 2).min(target_w);
        }
        if axes.y() {
            let target_h = self.height_stack.pop().unwrap();
            self.height = (self.height * 2).min(target_h);
        }
        (self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use crate::filter::gaussian_blur::{
        DecimationAxes, DecimationSizer, GaussianBlur, compute_gaussian_kernel, decimation_levels,
        plan_decimated_blur, transform_blur_params,
    };
    use crate::filter_effects::EdgeMode;
    use crate::kurbo::Affine;
    use alloc::vec::Vec;

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

    #[test]
    fn test_decimation_sizer_even() {
        let mut sizer = DecimationSizer::new(8, 8);
        assert_eq!(sizer.current(), (8, 8));

        assert_eq!(sizer.downscale(), (4, 4));
        assert_eq!(sizer.downscale(), (2, 2));

        assert_eq!(sizer.upscale(), (4, 4));
        assert_eq!(sizer.upscale(), (8, 8));
    }

    #[test]
    fn test_decimation_sizer_odd() {
        let mut sizer = DecimationSizer::new(5, 7);
        assert_eq!(sizer.downscale(), (3, 4));
        assert_eq!(sizer.downscale(), (2, 2));

        // Upscale clamps to the pre-downscale target
        assert_eq!(sizer.upscale(), (3, 4));
        assert_eq!(sizer.upscale(), (5, 7));
    }

    #[test]
    fn test_decimation_sizer_single_level() {
        let mut sizer = DecimationSizer::new(100, 50);
        assert_eq!(sizer.downscale(), (50, 25));
        assert_eq!(sizer.upscale(), (100, 50));
    }

    #[test]
    fn test_decimation_sizer_mixed_axes() {
        let mut sizer = DecimationSizer::new(100, 50);
        assert_eq!(sizer.downscale_axes(DecimationAxes::Both), (50, 25));
        assert_eq!(sizer.downscale_axes(DecimationAxes::X), (25, 25));
        assert_eq!(sizer.downscale_axes(DecimationAxes::X), (13, 25));

        assert_eq!(sizer.upscale_axes(DecimationAxes::X), (25, 25));
        assert_eq!(sizer.upscale_axes(DecimationAxes::X), (50, 25));
        assert_eq!(sizer.upscale_axes(DecimationAxes::Both), (100, 50));
    }

    #[test]
    fn test_decimation_levels_share_the_finest_levels() {
        let levels: Vec<_> = decimation_levels(3, 1).collect();
        assert_eq!(
            levels,
            [DecimationAxes::Both, DecimationAxes::X, DecimationAxes::X]
        );

        let levels: Vec<_> = decimation_levels(0, 2).collect();
        assert_eq!(levels, [DecimationAxes::Y, DecimationAxes::Y]);

        assert_eq!(decimation_levels(0, 0).len(), 0);
    }

    #[test]
    fn test_anisotropic_blur_plans_each_axis() {
        let blur = GaussianBlur::new(5.0, 0.0, EdgeMode::None);
        assert_eq!(blur.n_decimations_x, 2);
        assert_eq!(blur.n_decimations_y, 0);
        assert_eq!(blur.kernel_y(), [1.0]);
        assert!(blur.kernel_x().len() > 1);
    }

    #[test]
    fn test_uniform_blur_transforms_like_before() {
        let transform = Affine::rotate(0.7) * Affine::scale_non_uniform(2.0, 0.5);
        let (x, y) = transform_blur_params(3.0, 3.0, &transform);
        assert_eq!(x, y);
        // The mean of the singular values 2.0 and 0.5.
        assert!((x - 3.75).abs() < 1e-5);
    }

    #[test]
    fn test_anisotropic_blur_scales_per_axis() {
        let (x, y) = transform_blur_params(4.0, 1.0, &Affine::scale_non_uniform(2.0, 3.0));
        assert!((x - 8.0).abs() < 1e-5);
        assert!((y - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_anisotropic_blur_rotated_by_a_quarter_turn_swaps_axes() {
        let (x, y) = transform_blur_params(4.0, 1.0, &Affine::rotate(core::f64::consts::FRAC_PI_2));
        assert!((x - 1.0).abs() < 1e-5);
        assert!((y - 4.0).abs() < 1e-5);
    }
}
