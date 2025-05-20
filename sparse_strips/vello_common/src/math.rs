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
    let x = x * core::f32::consts::FRAC_2_SQRT_PI;
    let xx = x * x;
    let x = x + (0.24295 + (0.03395 + 0.0104 * xx) * xx) * (x * xx);
    x / (1.0 + x * x).sqrt()
}

const SCALAR_NEARLY_ZERO: f32 = 1.0 / (1 << 12) as f32;

pub trait FloatExt: Sized + Sub<f32, Output = f32> {
    fn is_nearly_zero(self) -> bool {
        self.is_nearly_zero_within_tolerance(SCALAR_NEARLY_ZERO)
    }

    fn is_nearly_equal(self, other: f32) -> bool {
        (self - other).abs() <= SCALAR_NEARLY_ZERO
    }

    fn is_nearly_zero_within_tolerance(self, tolerance: f32) -> bool;
}

impl FloatExt for f32 {
    fn is_nearly_zero_within_tolerance(self, tolerance: f32) -> bool {
        debug_assert!(tolerance >= 0.0);
        self.abs() <= tolerance
    }
}
