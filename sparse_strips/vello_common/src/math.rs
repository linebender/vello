// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Mathematical helper functions.

use core::ops::Sub;
#[cfg(not(feature = "std"))]
use peniko::kurbo::common::FloatFuncs as _;

// See https://raphlinus.github.io/audio/2018/09/05/sigmoid.html for a little
// explanation of this approximation to the erf function.
/// Approximate the erf function.
pub fn compute_erf7(x: f32) -> f32 {
    // Clamp `x`, because for large `x` the terms here become `inf`, causing the result to be 0 or
    // `NaN`. This clamping doesn't lose any information, because `erf(±10) ≈ 1` well within `f64`
    // machine precision, let alone `f32`.
    let x = x.clamp(-10., 10.);
    let x = x * core::f32::consts::FRAC_2_SQRT_PI;
    let xx = x * x;
    let x = x + (0.24295 + (0.03395 + 0.0104 * xx) * xx) * (x * xx);
    x / (1.0 + x * x).sqrt()
}

// From <https://github.com/linebender/tiny-skia/blob/68b198a7210a6bbf752b43d6bc4db62445730313/path/src/scalar.rs#L12>
// Note: If this value changes, also update NEARLY_ZERO_TOLERANCE in render_strips.wgsl
// @see {@link https://github.com/linebender/vello/blob/58b80d660e2fc5aef3bd32b24af3f95e973aab95/sparse_strips/vello_sparse_shaders/shaders/render_strips.wgsl#L63}
const SCALAR_NEARLY_ZERO: f32 = 1.0 / (1 << 12) as f32;

/// A number of useful methods for f32 numbers.
pub trait FloatExt: Sized + Sub<f32, Output = f32> {
    /// Whether the number is approximately 0.
    fn is_nearly_zero(&self) -> bool {
        self.is_nearly_zero_within_tolerance(SCALAR_NEARLY_ZERO)
    }

    /// Whether the number is approximately 0, with a given tolerance.
    fn is_nearly_zero_within_tolerance(&self, tolerance: f32) -> bool;
}

impl FloatExt for f32 {
    fn is_nearly_zero_within_tolerance(&self, tolerance: f32) -> bool {
        debug_assert!(tolerance >= 0.0, "tolerance must be positive");

        self.abs() <= tolerance
    }
}
