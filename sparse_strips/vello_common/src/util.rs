// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Utility functions.

use fearless_simd::{Simd, f32x16, u8x16};

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
