// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Utility functions.

use fearless_simd::{Simd, SimdInto, f32x16, u8x16};

/// Convert f32x16 to u8x16.
#[inline(always)]
pub fn f32_to_u8<S: Simd>(val: f32x16<S>) -> u8x16<S> {
    let simd = val.simd;
    // Note that converting to u32 first using SIMD and then u8
    // is much faster than converting directly from f32 to u8.
    let converted = simd.cvt_u32_f32x16(val);

    // TODO: Maybe we can also do this using SIMD?
    [
        converted[0] as u8,
        converted[1] as u8,
        converted[2] as u8,
        converted[3] as u8,
        converted[4] as u8,
        converted[5] as u8,
        converted[6] as u8,
        converted[7] as u8,
        converted[8] as u8,
        converted[9] as u8,
        converted[10] as u8,
        converted[11] as u8,
        converted[12] as u8,
        converted[13] as u8,
        converted[14] as u8,
        converted[15] as u8,
    ]
    .simd_into(val.simd)
}
