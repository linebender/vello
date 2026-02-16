//! The drop shadow filter.

use crate::color::{AlphaColor, Srgb};
use crate::filter::gaussian_blur::{MAX_KERNEL_SIZE, plan_decimated_blur, transform_blur_params};
use crate::filter_effects::EdgeMode;
use crate::kurbo::{Affine, Vec2};

/// A drop shadow filter.
#[derive(Debug)]
pub struct DropShadow {
    /// The x-offset of the shadow.
    pub dx: f32,
    /// The y-offset of the shadow.
    pub dy: f32,
    /// The color of the shadow.
    pub color: AlphaColor<Srgb>,
    /// Standard deviation for the blur (for reference/debugging).
    pub std_deviation: f32,
    /// Edge mode for blur sampling.
    pub edge_mode: EdgeMode,
    /// Number of 2x2 decimation levels to use (0 means direct convolution).
    pub n_decimations: usize,
    /// Pre-computed Gaussian kernel weights for the reduced blur.
    /// Only the first `kernel_size` elements are valid.
    pub kernel: [f32; MAX_KERNEL_SIZE],
    /// Actual length of the kernel (kernel is padded to `MAX_KERNEL_SIZE`).
    pub kernel_size: u8,
}

impl DropShadow {
    /// Create a new drop shadow filter with the specified parameters.
    ///
    /// This precomputes the blur decimation plan and kernel for optimal performance.
    pub fn new(
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

/// Transform a drop shadow's offset and standard deviation using the affine transformation.
///
/// Applies the full linear transformation (rotation, scale, and shear) to the offset vector,
/// and scales the blur standard deviation uniformly.
///
/// # Arguments
/// * `dx` - Horizontal offset in user space
/// * `dy` - Vertical offset in user space
/// * `std_deviation` - Blur standard deviation in user space
/// * `transform` - The transformation matrix to apply
///
/// # Returns
/// A tuple of (`scaled_dx`, `scaled_dy`, `scaled_std_dev`) in device space
pub(crate) fn transform_shadow_params(
    dx: f32,
    dy: f32,
    std_deviation: f32,
    transform: &Affine,
) -> (f32, f32, f32) {
    // Transform the offset vector by the full transformation matrix
    // to correctly handle rotation, scale, and shear.
    // We use the linear part only (no translation) since this is a vector offset.
    let offset = Vec2::new(dx as f64, dy as f64);
    let [a, b, c, d, _, _] = transform.as_coeffs();
    let transformed_offset = Vec2::new(a * offset.x + c * offset.y, b * offset.x + d * offset.y);
    let scaled_dx = transformed_offset.x as f32;
    let scaled_dy = transformed_offset.y as f32;

    // Scale the blur radius uniformly
    let scaled_std_dev = transform_blur_params(std_deviation, transform);

    (scaled_dx, scaled_dy, scaled_std_dev)
}
