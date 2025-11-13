// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Gaussian blur filter implementation using multi-scale separable convolution.
//!
//! This implementation uses a multi-scale approach for efficient blurring:
//! - **Small blurs** (σ ≤ 2): Direct separable convolution at full resolution
//! - **Large blurs** (σ > 2): Iterative downsample → blur → upsample pyramid
//!
//! The algorithm automatically determines the optimal number of decimation levels
//! using variance analysis. Each 2× decimation applies a \[1,3,3,1\]/8 binomial filter
//! (adding variance = 3.0), then downsamples, reducing the remaining blur work needed.
//! This exploits the variance additivity property: `σ²_total = σ²_downsample + σ²_blur`.

use super::FilterEffect;
use crate::layer_manager::LayerManager;
use alloc::vec;
use core::f32::consts::E;
use vello_common::filter_effects::EdgeMode;
use vello_common::peniko::color::PremulRgba8;
#[cfg(not(feature = "std"))]
use vello_common::peniko::kurbo::common::FloatFuncs as _;
use vello_common::pixmap::Pixmap;

/// Maximum size of the Gaussian kernel (must be odd).
/// Kernels larger than this are truncated. In practice, large blurs use multi-scale
/// decimation, so this limit only applies to the reduced blur at coarsest resolution.
pub(crate) const MAX_KERNEL_SIZE: usize = 17;

pub(crate) struct GaussianBlur {
    std_deviation: f32,
    /// Number of 2× decimation levels to use (0 means no decimation, direct convolution).
    n_decimations: usize,
    /// Pre-computed Gaussian kernel weights for the reduced blur.
    /// Only the first `kernel_size` elements are valid.
    kernel: [f32; MAX_KERNEL_SIZE],
    /// Actual length of the kernel (rest is padding up to `MAX_KERNEL_SIZE`).
    kernel_size: usize,
    /// Edge mode for handling out-of-bounds sampling.
    edge_mode: EdgeMode,
}

impl GaussianBlur {
    /// Create a new Gaussian blur filter with the specified standard deviation.
    ///
    /// This precomputes the decimation plan, kernel, and radius for optimal performance.
    pub(crate) fn new(std_deviation: f32, edge_mode: EdgeMode) -> Self {
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

impl FilterEffect for GaussianBlur {
    fn execute_lowp(&self, pixmap: &mut Pixmap, layer_manager: &mut LayerManager) {
        // No blur if std_deviation is zero or negative
        if self.std_deviation <= 0.0 {
            return;
        }

        let scratch = layer_manager.get_scratch_buffer(pixmap.width(), pixmap.height());
        apply_blur(
            pixmap,
            scratch,
            self.n_decimations,
            &self.kernel[..self.kernel_size],
            self.edge_mode,
        );
    }

    fn execute_highp(&self, pixmap: &mut Pixmap, layer_manager: &mut LayerManager) {
        // TODO: Currently only lowp is implemented and used for highp as well.
        // This needs to be updated to use proper high-precision arithmetic.

        // No blur if std_deviation is zero or negative
        if self.std_deviation <= 0.0 {
            return;
        }

        let scratch = layer_manager.get_scratch_buffer(pixmap.width(), pixmap.height());
        apply_blur(
            pixmap,
            scratch,
            self.n_decimations,
            &self.kernel[..self.kernel_size],
            self.edge_mode,
        );
    }
}

/// Compute the blur execution plan based on standard deviation.
///
/// Returns (`n_decimations`, `kernel`, `kernel_size`):
/// - `n_decimations`: Number of 2× downsampling steps to perform (per axis)
/// - `kernel`: Pre-computed Gaussian kernel weights (fixed-size array)
/// - `kernel_size`: Actual length of the kernel (rest is zero-padded)
pub(crate) fn plan_decimated_blur(std_deviation: f32) -> (usize, [f32; MAX_KERNEL_SIZE], usize) {
    if std_deviation <= 0.0 {
        // Invalid standard deviation, return identity kernel (no blur)
        let mut kernel = [0.0; MAX_KERNEL_SIZE];
        kernel[0] = 1.0;
        return (0, kernel, 1);
    }

    // Compute decimation plan using variance analysis.
    // Variance (σ²) has the additive property: applying two blurs sequentially
    // adds their variances together. We use this to decompose the blur.
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
pub(crate) fn compute_gaussian_kernel(std_deviation: f32) -> ([f32; MAX_KERNEL_SIZE], usize) {
    // Use radius = 3σ to capture 99.7% of the Gaussian distribution.
    // Beyond ±3σ, the Gaussian values are negligible (<0.3%).
    let radius = (3.0 * std_deviation).ceil() as usize;
    let kernel_size = 1 + radius * 2;
    let kernel_size = kernel_size.min(MAX_KERNEL_SIZE);

    let mut kernel = [0.0; MAX_KERNEL_SIZE];
    // Compute Gaussian weights using the formula: G(x) = exp(-x² / (2σ²))
    // This creates a symmetric bell curve centered at the middle of the kernel.
    let gaussian_denominator = 2.0 * std_deviation * std_deviation;
    let mut sum = 0.0;
    let kernel_center = (kernel_size / 2) as f32;
    for (i, weight) in kernel.iter_mut().enumerate().take(kernel_size) {
        // Compute distance from center (0 at center, increases outward)
        let x = (i as f32) - kernel_center;
        // Apply Gaussian formula: weight decreases exponentially with squared distance
        *weight = E.powf(-x * x / gaussian_denominator);
        sum += *weight;
    }

    // Normalize weights to sum to 1.0, ensuring the blur doesn't change overall brightness.
    // Without normalization, blurring a uniform gray area could make it brighter/darker.
    let scale = 1.0 / sum;
    for weight in kernel.iter_mut().take(kernel_size) {
        *weight *= scale;
    }

    (kernel, kernel_size)
}

/// Apply Gaussian blur using multi-scale decimation and upsampling.
///
/// Uses a precomputed decimation plan and kernel for optimal performance.
/// Operates in-place using a single pixmap buffer with logical dimension tracking
/// to minimize memory allocations. For `n_decimations=0`, applies direct convolution.
///
/// The `scratch` buffer is used for separable convolution and must be at least as
/// large as the source pixmap.
pub(crate) fn apply_blur(
    pixmap: &mut Pixmap,
    scratch: &mut Pixmap,
    n_decimations: usize,
    kernel: &[f32],
    edge_mode: EdgeMode,
) {
    let radius = kernel.len() / 2;
    let width = pixmap.width();
    let height = pixmap.height();

    // Small blur: apply direct convolution at full resolution
    if n_decimations == 0 {
        convolve(pixmap, scratch, width, height, kernel, radius, edge_mode);
        return;
    }

    // Track logical dimensions through decimation (physical buffer stays the same size)
    let mut src_width = width;
    let mut src_height = height;
    let mut dimensions_stack = vec![(width, height)];

    // Downsample n times (each step reduces resolution by 2×)
    for _ in 0..n_decimations {
        (src_width, src_height) = downscale(pixmap, src_width, src_height, edge_mode);
        dimensions_stack.push((src_width, src_height));
    }

    // Apply the reduced blur at the coarsest resolution
    convolve(
        pixmap, scratch, src_width, src_height, kernel, radius, edge_mode,
    );

    // Upsample back to original resolution (each step doubles resolution by 2×)
    for _ in 0..n_decimations {
        dimensions_stack.pop();
        if let Some(&(target_width, target_height)) = dimensions_stack.last() {
            (src_width, src_height) = upscale(pixmap, src_width, src_height, edge_mode);
            // Clamp because upscale can exceed target on odd dimensions (e.g., 5→3→6 > 5)
            src_width = src_width.min(target_width);
            src_height = src_height.min(target_height);
        }
    }

    // Final dimensions should match original
    debug_assert_eq!(
        (src_width, src_height),
        (width, height),
        "Final dimensions should match original"
    );
}

/// Apply separable Gaussian convolution with logical dimensions.
///
/// Performs horizontal blur followed by vertical blur. Works with a logical view
/// of the pixmap, using only the top-left region defined by width × height.
/// The `temp` buffer is provided by the caller to avoid allocations.
pub(crate) fn convolve(
    src: &mut Pixmap,
    scratch: &mut Pixmap,
    width: u16,
    height: u16,
    kernel: &[f32],
    radius: usize,
    edge_mode: EdgeMode,
) {
    convolve_x(src, scratch, width, height, kernel, radius, edge_mode);
    convolve_y(scratch, src, width, height, kernel, radius, edge_mode);
}

/// Apply horizontal blur pass (1D convolution along x-axis).
///
/// For each output pixel, computes a weighted sum of horizontally neighboring pixels
/// using the Gaussian kernel. Handles edge cases according to the specified edge mode.
/// Writes results to a destination buffer to avoid overwriting source data.
pub(crate) fn convolve_x(
    src: &Pixmap,
    dst: &mut Pixmap,
    src_width: u16,
    src_height: u16,
    kernel: &[f32],
    radius: usize,
    edge_mode: EdgeMode,
) {
    for y in 0..src_height {
        for x in 0..src_width {
            let mut rgba = [0.0_f32; 4];

            // Sum contributions from all kernel positions: output = Σ(weight[j] × pixel[x+j-radius])
            for (j, &k) in kernel.iter().enumerate() {
                let src_x = x as i32 + j as i32 - radius as i32;
                let p = sample_x(src, src_x, y, src_width, edge_mode);

                rgba[0] += p.r as f32 * k;
                rgba[1] += p.g as f32 * k;
                rgba[2] += p.b as f32 * k;
                rgba[3] += p.a as f32 * k;
            }

            // Convert back to u8 with rounding
            dst.set_pixel(
                x,
                y,
                PremulRgba8 {
                    r: rgba[0].round() as u8,
                    g: rgba[1].round() as u8,
                    b: rgba[2].round() as u8,
                    a: rgba[3].round() as u8,
                },
            );
        }
    }
}

/// Apply vertical blur pass (1D convolution along y-axis).
///
/// For each output pixel, computes a weighted sum of vertically neighboring pixels
/// using the Gaussian kernel. Handles edge cases according to the specified edge mode.
/// Writes results to a destination buffer to avoid overwriting source data.
pub(crate) fn convolve_y(
    src: &Pixmap,
    dst: &mut Pixmap,
    src_width: u16,
    src_height: u16,
    kernel: &[f32],
    radius: usize,
    edge_mode: EdgeMode,
) {
    for y in 0..src_height {
        for x in 0..src_width {
            let mut rgba = [0.0_f32; 4];

            // Sum contributions from all kernel positions: output = Σ(weight[j] × pixel[y+j-radius])
            for (j, &k) in kernel.iter().enumerate() {
                let src_y = y as i32 + j as i32 - radius as i32;
                let p = sample_y(src, x, src_y, src_height, edge_mode);

                rgba[0] += p.r as f32 * k;
                rgba[1] += p.g as f32 * k;
                rgba[2] += p.b as f32 * k;
                rgba[3] += p.a as f32 * k;
            }

            // Convert back to u8 with rounding
            dst.set_pixel(
                x,
                y,
                PremulRgba8 {
                    r: rgba[0].round() as u8,
                    g: rgba[1].round() as u8,
                    b: rgba[2].round() as u8,
                    a: rgba[3].round() as u8,
                },
            );
        }
    }
}

/// Downsample image by 2x using separable \[1,3,3,1\]/8 binomial filter.
///
/// Performs horizontal and vertical decimation in sequence. Returns the new
/// logical dimensions (ceil(width/2), ceil(height/2)).
pub(crate) fn downscale(
    src: &mut Pixmap,
    src_width: u16,
    src_height: u16,
    edge_mode: EdgeMode,
) -> (u16, u16) {
    let dst_width = src_width.div_ceil(2);
    let dst_height = src_height.div_ceil(2);
    downscale_x(src, src_width, src_height, dst_width, edge_mode);
    downscale_y(src, src_width, src_height, dst_height, edge_mode);
    (dst_width, dst_height)
}

/// Horizontal decimation pass using \[1,3,3,1\]/8 filter.
///
/// Reduces width by 2x while applying a binomial blur kernel. The \[1,3,3,1\] weights
/// approximate a Gaussian and contribute variance=0.75 before the 2x downsampling.
fn downscale_x(
    src: &mut Pixmap,
    src_width: u16,
    src_height: u16,
    dst_width: u16,
    edge_mode: EdgeMode,
) {
    for y in 0..src_height {
        // Sliding window: maintains 2 previous pixels (x0, x1) to form 4-tap filter
        // with current pixels (x2, x3). Start with sample at -1 for implicit left padding.
        let mut p0 = sample_x(src, -1, y, src_width, edge_mode);
        let mut p1 = sample_x(src, 0, y, src_width, edge_mode);

        for x in 0..dst_width {
            // Sample 4 horizontally adjacent pixels for [1,3,3,1] kernel
            // Pattern: [x*2-1, x*2, x*2+1, x*2+2]
            let src_x = (x * 2) as i32;
            let p2 = sample_x(src, src_x + 1, y, src_width, edge_mode);
            let p3 = sample_x(src, src_x + 2, y, src_width, edge_mode);

            // Apply [1,3,3,1]/8 weights → output = (p0 + 3×p1 + 3×p2 + p3) / 8
            src.set_pixel(x, y, decimate_weighted(p0, p1, p2, p3));

            // Advance window: previous p2,p3 become next p0,p1
            p0 = p2;
            p1 = p3;
        }
    }
}

/// Vertical decimation pass using \[1,3,3,1\]/8 filter.
///
/// Reduces logical height by 2x while applying a binomial blur kernel.
/// Operates in-place by writing to the beginning of the same buffer.
fn downscale_y(
    src: &mut Pixmap,
    src_width: u16,
    src_height: u16,
    dst_height: u16,
    edge_mode: EdgeMode,
) {
    for x in 0..src_width {
        // Sliding window: maintains 2 previous pixels (y0, y1) to form 4-tap filter
        // with current pixels (y2, y3). Start with sample at -1 for implicit top padding.
        let mut p0 = sample_y(src, x, -1, src_height, edge_mode);
        let mut p1 = sample_y(src, x, 0, src_height, edge_mode);

        for y in 0..dst_height {
            // Sample 4 vertically adjacent pixels for [1,3,3,1] kernel
            // Pattern: [y*2-1, y*2, y*2+1, y*2+2]
            let src_y = (y * 2) as i32;
            let p2 = sample_y(src, x, src_y + 1, src_height, edge_mode);
            let p3 = sample_y(src, x, src_y + 2, src_height, edge_mode);

            // Apply [1,3,3,1]/8 weights → output = (p0 + 3×p1 + 3×p2 + p3) / 8
            src.set_pixel(x, y, decimate_weighted(p0, p1, p2, p3));

            // Advance window: previous p2,p3 become next p0,p1
            p0 = p2;
            p1 = p3;
        }
    }
}

/// Upsample a pixmap by 2x using linear interpolation with [0.75, 0.25] weights.
///
/// Uses separable passes: horizontal doubling followed by vertical doubling.
///
/// ## Phase Alignment Theory
///
/// The downsampling \[1,3,3,1\]/8 filter creates a half-pixel offset. Its center of mass
/// is at position `2k + 0.5`, not `2k`. This means each downsampled pixel at discrete
/// position `k` represents a sample at continuous position `2k + 0.5`.
///
/// When upsampling, we reconstruct pixels at positions `2k` and `2k+1` from downsampled
/// pixels whose centers are at `..., 2k-1.5, 2k+0.5, 2k+2.5, ...`
///
/// **Interpolation weights** are derived from linear interpolation based on distances:
/// - Position `2k`:   distance 0.5 from center at `2k+0.5`, distance 1.5 from center at `2k-1.5`
///   → weights: 0.75×pixel\[k\] + 0.25×pixel\[k-1\]
/// - Position `2k+1`: distance 0.5 from center at `2k+0.5`, distance 1.5 from center at `2k+2.5`
///   → weights: 0.75×pixel\[k\] + 0.25×pixel\[k+1\]
pub(crate) fn upscale(
    src: &mut Pixmap,
    src_width: u16,
    src_height: u16,
    edge_mode: EdgeMode,
) -> (u16, u16) {
    let dst_width = src_width * 2;
    let dst_height = src_height * 2;
    upscale_x(src, src_width, src_height, edge_mode);
    upscale_y(src, src_width, src_height, edge_mode);
    (dst_width, dst_height)
}

/// Horizontal upsampling pass using [0.75, 0.25] interpolation with logical dimensions.
///
/// Doubles the logical width using phase-aligned interpolation. Each input pixel
/// generates two output pixels with different weights based on their distance from
/// the downsampled pixel's center position.
/// Operates in-place by processing backwards to avoid overwriting source data.
fn upscale_x(src: &mut Pixmap, src_width: u16, src_height: u16, edge_mode: EdgeMode) {
    // Process backwards (right to left) to avoid overwriting source data
    for y in 0..src_height {
        // Maintain sliding window of three pixels: prev, current, next
        // This allows us to compute both output pixels that depend on current pixel x
        let mut p0 = sample_x(src, src_width as i32 + 1, y, src_width, edge_mode);
        let mut p1 = sample_x(src, src_width as i32, y, src_width, edge_mode);

        for x in (0..src_width).rev() {
            let src_x = x as i32;
            let p2 = sample_x(src, src_x - 1, y, src_width, edge_mode);

            // Generate two output pixels per input with phase-aligned interpolation:
            // output[2x]   = 0.25×p2 + 0.75×p1  (position 2x   is 0.5 from center at 2x+0.5)
            // output[2x+1] = 0.75×p1 + 0.25×p0  (position 2x+1 is 0.5 from center at 2x+0.5)
            let dst_x = x * 2;
            src.set_pixel(dst_x, y, interpolate_25_75(p2, p1));
            src.set_pixel(dst_x + 1, y, interpolate_75_25(p1, p0));

            // Advance sliding window for next iteration
            p0 = p1;
            p1 = p2;
        }
    }
}

/// Vertical upsampling pass using [0.75, 0.25] interpolation with logical dimensions.
///
/// Doubles the logical height using phase-aligned interpolation. Each input pixel
/// generates two output pixels with different weights based on their distance from
/// the downsampled pixel's center position.
/// Operates in-place by processing backwards to avoid overwriting source data.
fn upscale_y(src: &mut Pixmap, src_width: u16, src_height: u16, edge_mode: EdgeMode) {
    // Process backwards (bottom to top) to avoid overwriting source data
    for x in 0..src_width {
        // Maintain sliding window of three pixels: prev, current, next
        // This allows us to compute both output pixels that depend on current pixel y
        let mut p0 = sample_y(src, x, src_height as i32 + 1, src_height, edge_mode);
        let mut p1 = sample_y(src, x, src_height as i32, src_height, edge_mode);

        for y in (0..src_height).rev() {
            let src_y = y as i32;
            let p2 = sample_y(src, x, src_y - 1, src_height, edge_mode);

            // Generate two output rows per input with phase-aligned interpolation:
            // output[2y]   = 0.25×p2 + 0.75×p1  (position 2y   is 0.5 from center at 2y+0.5)
            // output[2y+1] = 0.75×p1 + 0.25×p0  (position 2y+1 is 0.5 from center at 2y+0.5)
            let dst_y = y * 2;
            src.set_pixel(x, dst_y, interpolate_25_75(p2, p1));
            src.set_pixel(x, dst_y + 1, interpolate_75_25(p1, p0));

            // Advance sliding window for next iteration
            p0 = p1;
            p1 = p2;
        }
    }
}

/// Transparent black pixel constant.
const TRANSPARENT_BLACK: PremulRgba8 = PremulRgba8 {
    r: 0,
    g: 0,
    b: 0,
    a: 0,
};

/// Sample a pixel with edge mode handling for horizontal sampling.
#[inline(always)]
fn sample_x(src: &Pixmap, x: i32, y: u16, width: u16, edge_mode: EdgeMode) -> PremulRgba8 {
    sample(x, width, edge_mode, |src_x| src.sample(src_x, y))
}

/// Sample a pixel with edge mode handling for vertical sampling.
#[inline(always)]
fn sample_y(src: &Pixmap, x: u16, y: i32, height: u16, edge_mode: EdgeMode) -> PremulRgba8 {
    sample(y, height, edge_mode, |src_y| src.sample(x, src_y))
}

/// Sample a pixel with edge mode handling (generic implementation).
///
/// Handles both horizontal and vertical sampling based on the provided closure.
/// The `sample_fn` closure receives the clamped/extended coordinate and returns the pixel.
/// For `EdgeMode::None`, returns transparent black if the coordinate is out of bounds.
#[inline(always)]
fn sample<F>(coord: i32, size: u16, edge_mode: EdgeMode, sample_fn: F) -> PremulRgba8
where
    F: FnOnce(u16) -> PremulRgba8,
{
    // For EdgeMode::None, return transparent black if out of bounds
    if edge_mode == EdgeMode::None && (coord < 0 || coord >= size as i32) {
        return TRANSPARENT_BLACK;
    }
    let extended_coord = extend(coord, size, edge_mode);
    sample_fn(extended_coord)
}

/// Extend a coordinate beyond image boundaries according to the edge mode.
///
/// Transforms out-of-bounds coordinates for sampling: clamped, wrapped, or mirrored
/// depending on the mode. For `EdgeMode::None`, the coordinate is guaranteed to be
/// in-bounds (already checked by caller, which returns transparent black for out-of-bounds).
#[inline(always)]
fn extend(coord: i32, size: u16, edge_mode: EdgeMode) -> u16 {
    match edge_mode {
        EdgeMode::Duplicate => {
            // Clamp to image bounds: pixels outside use nearest edge pixel
            coord.clamp(0, size as i32 - 1) as u16
        }
        EdgeMode::None => {
            // Coordinate is already validated as in-bounds by caller
            coord as u16
        }
        EdgeMode::Wrap => {
            // Wrap around using modulo: image tiles infinitely
            let mut c = coord % size as i32;
            if c < 0 {
                c += size as i32;
            }
            c as u16
        }
        EdgeMode::Mirror => {
            // Mirror at boundaries: image reflects across edges
            let period = size as i32 * 2;
            let mut c = coord % period;
            if c < 0 {
                c += period;
            }
            if c >= size as i32 {
                c = period - c - 1;
            }
            c as u16
        }
    }
}

/// Blend 4 RGBA pixels using \[1,3,3,1\]/8 binomial weights.
///
/// Computes `(p0 + 3×p1 + 3×p2 + p3) / 8` using efficient integer arithmetic (right shift).
/// This binomial pattern approximates a Gaussian and contributes variance=0.75 to the blur.
#[inline(always)]
fn decimate_weighted(
    p0: PremulRgba8,
    p1: PremulRgba8,
    p2: PremulRgba8,
    p3: PremulRgba8,
) -> PremulRgba8 {
    let r = ((p0.r as u32 + p1.r as u32 * 3 + p2.r as u32 * 3 + p3.r as u32) >> 3) as u8;
    let g = ((p0.g as u32 + p1.g as u32 * 3 + p2.g as u32 * 3 + p3.g as u32) >> 3) as u8;
    let b = ((p0.b as u32 + p1.b as u32 * 3 + p2.b as u32 * 3 + p3.b as u32) >> 3) as u8;
    let a = ((p0.a as u32 + p1.a as u32 * 3 + p2.a as u32 * 3 + p3.a as u32) >> 3) as u8;
    PremulRgba8 { r, g, b, a }
}

/// Blend 2 RGBA pixels using [0.25, 0.75] weights (right-weighted interpolation).
///
/// Used for upsampling to generate the second of two output pixels.
/// Favors the right/next pixel (p1) with 75% weight.
#[inline(always)]
fn interpolate_25_75(p0: PremulRgba8, p1: PremulRgba8) -> PremulRgba8 {
    let r = ((p0.r as u32 + p1.r as u32 * 3) >> 2) as u8;
    let g = ((p0.g as u32 + p1.g as u32 * 3) >> 2) as u8;
    let b = ((p0.b as u32 + p1.b as u32 * 3) >> 2) as u8;
    let a = ((p0.a as u32 + p1.a as u32 * 3) >> 2) as u8;
    PremulRgba8 { r, g, b, a }
}

/// Blend 2 RGBA pixels using [0.75, 0.25] weights.
///
/// Computes `0.75×p0 + 0.25×p1` using efficient integer arithmetic.
/// Used during upsampling for positions closer to the first pixel.
#[inline(always)]
fn interpolate_75_25(p0: PremulRgba8, p1: PremulRgba8) -> PremulRgba8 {
    let r = ((p0.r as u32 * 3 + p1.r as u32) >> 2) as u8;
    let g = ((p0.g as u32 * 3 + p1.g as u32) >> 2) as u8;
    let b = ((p0.b as u32 * 3 + p1.b as u32) >> 2) as u8;
    let a = ((p0.a as u32 * 3 + p1.a as u32) >> 2) as u8;
    PremulRgba8 { r, g, b, a }
}
