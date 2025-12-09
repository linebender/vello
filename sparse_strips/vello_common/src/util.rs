// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Utility functions.

use fearless_simd::{
    Bytes, Simd, SimdBase, SimdFloat, f32x16, u8x16, u8x32, u16x16, u16x32, u32x16,
};
use peniko::kurbo::Affine;
#[cfg(not(feature = "std"))]
use peniko::kurbo::common::FloatFuncs as _;

/// Convert f32x16 to u8x16.
#[inline(always)]
pub fn f32_to_u8<S: Simd>(val: f32x16<S>) -> u8x16<S> {
    let simd = val.simd;
    let converted = val.to_int::<u32x16<S>>().to_bytes();

    let (x8_1, x8_2) = simd.split_u8x64(converted);
    let (p1, p2) = simd.split_u8x32(x8_1);
    let (p3, p4) = simd.split_u8x32(x8_2);

    let uzp1 = simd.unzip_low_u8x16(p1, p2);
    let uzp2 = simd.unzip_low_u8x16(p3, p4);
    simd.unzip_low_u8x16(uzp1, uzp2)
}

/// A trait for implementing a fast approximal division by 255 for integers.
pub trait Div255Ext {
    /// Divide by 255.
    fn div_255(self) -> Self;
}

impl<S: Simd> Div255Ext for u16x32<S> {
    #[inline(always)]
    fn div_255(self) -> Self {
        let p1 = Self::splat(self.simd, 255);
        let p2 = self + p1;
        p2 >> 8
    }
}

impl<S: Simd> Div255Ext for u16x16<S> {
    #[inline(always)]
    fn div_255(self) -> Self {
        let p1 = Self::splat(self.simd, 255);
        let p2 = self + p1;
        p2 >> 8
    }
}

/// Perform a normalized multiplication for u8x32.
#[inline(always)]
pub fn normalized_mul_u8x32<S: Simd>(a: u8x32<S>, b: u8x32<S>) -> u16x32<S> {
    (S::widen_u8x32(a.simd, a) * S::widen_u8x32(b.simd, b)).div_255()
}

/// Perform a normalized multiplication for u8x16.
#[inline(always)]
pub fn normalized_mul_u8x16<S: Simd>(a: u8x16<S>, b: u8x16<S>) -> u16x16<S> {
    (S::widen_u8x16(a.simd, a) * S::widen_u8x16(b.simd, b)).div_255()
}

/// Extract scale factors from an affine transform using singular value decomposition.
///
/// Returns a tuple of (`scale_x`, `scale_y`) representing the scale along each axis.
/// This uses the same algorithm as kurbo's internal `svd()` method.
///
/// # Arguments
/// * `transform` - The affine transformation to extract scales from.
///
/// # Returns
/// A tuple `(scale_x, scale_y)` with minimum values clamped to 1e-6 to avoid division by zero.
///
/// # Note
/// TODO: Consider making `Affine::svd()` public in kurbo to avoid duplicating this code.
/// This implementation mirrors kurbo's internal SVD calculation for extracting scale factors
/// from arbitrary affine transformations.
#[inline]
pub fn extract_scales(transform: &Affine) -> (f32, f32) {
    let [a, b, c, d, _, _] = transform.as_coeffs();
    let a = a as f32;
    let b = b as f32;
    let c = c as f32;
    let d = d as f32;

    // Compute singular values using the same formula as kurbo's svd()
    let a2 = a * a;
    let b2 = b * b;
    let c2 = c * c;
    let d2 = d * d;
    let s1 = a2 + b2 + c2 + d2;
    let s2 = ((a2 - b2 + c2 - d2).powi(2) + 4.0 * (a * b + c * d).powi(2)).sqrt();

    let scale_x = (0.5 * (s1 + s2)).sqrt();
    let scale_y = (0.5 * (s1 - s2)).sqrt();

    (scale_x.max(1e-6), scale_y.max(1e-6))
}
