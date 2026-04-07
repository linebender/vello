// Copyright 2025 the Vello Authors and the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Utility helper functions.

use core::ops::Sub;
use peniko::kurbo::Affine;

// From <https://github.com/linebender/tiny-skia/blob/68b198a7210a6bbf752b43d6bc4db62445730313/path/src/scalar.rs#L12>
const SCALAR_NEARLY_ZERO_F32: f32 = 1.0 / (1 << 12) as f32;
const SCALAR_NEARLY_ZERO_F64: f64 = 1.0 / (1 << 12) as f64;

/// A number of useful methods for f32 numbers.
pub(crate) trait FloatExt: Sized + Sub<f32, Output = f32> {
    /// Whether the number is approximately 0.
    fn is_nearly_zero(&self) -> bool {
        self.is_nearly_zero_within_tolerance(SCALAR_NEARLY_ZERO_F32)
    }

    /// Whether the number is approximately 0, with a given tolerance.
    fn is_nearly_zero_within_tolerance(&self, tolerance: f32) -> bool;
}

impl FloatExt for f32 {
    #[inline(always)]
    fn is_nearly_zero_within_tolerance(&self, tolerance: f32) -> bool {
        debug_assert!(tolerance >= 0.0, "tolerance must be positive");

        self.abs() <= tolerance
    }
}

pub(crate) trait AffineExt {
    /// Whether the transform has any skewing coefficient.
    fn has_skew(&self) -> bool;

    /// Whether the transform has a scaling factor not equal to 1 or -1.
    #[cfg(any(feature = "vello_cpu", feature = "vello_hybrid"))]
    fn has_non_unit_scale(&self) -> bool;

    /// Whether the transform has a vertical skew.
    fn has_vertical_skew(&self) -> bool;

    /// Whether the transform has positive, uniform scaling factors and no skew.
    fn is_positive_uniform_scale_without_skew(&self) -> bool;

    /// Whether the transform has positive, uniform scaling factors and no vertical skew.
    fn is_positive_uniform_scale_without_vertical_skew(&self) -> bool;
}

impl AffineExt for Affine {
    #[inline]
    fn has_skew(&self) -> bool {
        let [_, b, c, _, _, _] = self.as_coeffs();
        b.abs() > SCALAR_NEARLY_ZERO_F64 || c.abs() > SCALAR_NEARLY_ZERO_F64
    }

    #[cfg(any(feature = "vello_cpu", feature = "vello_hybrid"))]
    #[inline]
    fn has_non_unit_scale(&self) -> bool {
        let [a, _, _, d, _, _] = self.as_coeffs();
        (a.abs() - 1.0).abs() > SCALAR_NEARLY_ZERO_F64
            || (d.abs() - 1.0).abs() > SCALAR_NEARLY_ZERO_F64
    }

    #[inline]
    fn is_positive_uniform_scale_without_skew(&self) -> bool {
        let [a, _, _, d, _, _] = self.as_coeffs();
        (a - d).abs() <= SCALAR_NEARLY_ZERO_F64 && a > 0.0 && d > 0.0 && !self.has_skew()
    }

    #[inline]
    fn has_vertical_skew(&self) -> bool {
        let [_, b, _, _, _, _] = self.as_coeffs();
        b.abs() > SCALAR_NEARLY_ZERO_F64
    }

    #[inline]
    fn is_positive_uniform_scale_without_vertical_skew(&self) -> bool {
        let [a, _, _, d, _, _] = self.as_coeffs();
        (a - d).abs() <= SCALAR_NEARLY_ZERO_F64 && a > 0.0 && d > 0.0 && !self.has_vertical_skew()
    }
}
