// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Utility functions.

use fearless_simd::{Simd, SimdBase, f32x16, u8x16, u8x32, u16x16, u16x32};

/// Convert f32x16 to u8x16.
#[inline(always)]
pub fn f32_to_u8<S: Simd>(val: f32x16<S>) -> u8x16<S> {
    let simd = val.simd;
    let converted = val.cvt_u32().reinterpret_u8();

    let (x8_1, x8_2) = simd.split_u8x64(converted);
    let (p1, p2) = simd.split_u8x32(x8_1);
    let (p3, p4) = simd.split_u8x32(x8_2);

    let uzp1 = simd.unzip_low_u8x16(p1, p2);
    let uzp2 = simd.unzip_low_u8x16(p3, p4);
    simd.unzip_low_u8x16(uzp1, uzp2)
}

pub trait Div255Ext {
    fn div_255(self) -> Self;
}

impl<S: Simd> Div255Ext for u16x32<S> {
    #[inline(always)]
    fn div_255(self) -> Self {
        let p1 = Self::splat(self.simd, 255);
        let p2 = self + p1;
        p2.shr(8)
    }
}

impl<S: Simd> Div255Ext for u16x16<S> {
    #[inline(always)]
    fn div_255(self) -> Self {
        let p1 = Self::splat(self.simd, 255);
        let p2 = self + p1;
        p2.shr(8)
    }
}

#[inline(always)]
pub fn normalized_mul_u8x32<S: Simd>(a: u8x32<S>, b: u8x32<S>) -> u16x32<S> {
    (S::widen_u8x32(a.simd, a) * S::widen_u8x32(b.simd, b)).div_255()
}

#[inline(always)]
pub fn normalized_mul_u8x16<S: Simd>(a: u8x16<S>, b: u8x16<S>) -> u16x16<S> {
    (S::widen_u8x16(a.simd, a) * S::widen_u8x16(b.simd, b)).div_255()
}
