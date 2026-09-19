// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A number of SIMD extension traits.

use fearless_simd::*;

/// Splatting every 4th element in the vector, used for splatting the alpha value of
/// a color to all lanes.
pub trait Splat4thExt<S> {
    /// Splat every 4th element of the vector.
    fn splat_4th(self) -> Self;
}

impl<S: Simd> Splat4thExt<S> for f32x4<S> {
    #[inline(always)]
    fn splat_4th(self) -> Self {
        // TODO: Explore whether it's just faster to manually access the 4th element and splat it.
        let zip1 = self.zip_high(self);
        zip1.zip_high(zip1)
    }
}

impl<S: Simd> Splat4thExt<S> for f32x8<S> {
    #[inline(always)]
    fn splat_4th(self) -> Self {
        let (mut p1, mut p2) = self.simd.split_f32x8(self);
        p1 = p1.splat_4th();
        p2 = p2.splat_4th();

        self.simd.combine_f32x4(p1, p2)
    }
}

impl<S: Simd> Splat4thExt<S> for f32x16<S> {
    #[inline(always)]
    fn splat_4th(self) -> Self {
        let (mut p1, mut p2) = self.simd.split_f32x16(self);
        p1 = p1.splat_4th();
        p2 = p2.splat_4th();

        self.simd.combine_f32x8(p1, p2)
    }
}

impl<S: Simd> Splat4thExt<S> for u8x16<S> {
    #[inline(always)]
    fn splat_4th(self) -> Self {
        // TODO: SIMDify
        [
            self[3], self[3], self[3], self[3], self[7], self[7], self[7], self[7], self[11],
            self[11], self[11], self[11], self[15], self[15], self[15], self[15],
        ]
        .simd_into(self.simd)
    }
}

impl<S: Simd> Splat4thExt<S> for u8x32<S> {
    #[inline(always)]
    fn splat_4th(self) -> Self {
        let (mut p1, mut p2) = self.simd.split_u8x32(self);
        p1 = p1.splat_4th();
        p2 = p2.splat_4th();

        self.simd.combine_u8x16(p1, p2)
    }
}

/// Splat each single element in the vector to 4 lanes.
#[inline(always)]
pub fn element_wise_splat<S: Simd>(simd: S, input: f32x4<S>) -> f32x16<S> {
    simd.combine_f32x8(
        simd.combine_f32x4(f32x4::splat(simd, input[0]), f32x4::splat(simd, input[1])),
        simd.combine_f32x4(f32x4::splat(simd, input[2]), f32x4::splat(simd, input[3])),
    )
}
