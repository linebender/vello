// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::fine::common::gradient::apply_extend;
use core::slice::Iter;
use fearless_simd_macros::simd;
use vello_common::encode::EncodedGradient;
use vello_common::fearless_simd::*;

/// An accelerated gradient painter for u8.
///
/// Assumes that the gradient has no undefined positions.
#[derive(Debug)]
pub(crate) struct GradientPainter<'a, S: Simd> {
    gradient: &'a EncodedGradient,
    lut: &'a [[u8; 4]],
    t_vals: Iter<'a, [f32; 16]>,
    scale_factor: f32x16<S>,
    simd: S,
}

impl<'a, S: Simd> GradientPainter<'a, S> {
    #[simd]
    pub(crate) fn new(simd: S, gradient: &'a EncodedGradient, t_vals: &'a [f32]) -> Self {
        let lut = gradient.u8_lut(simd);
        let scale_factor = f32x16::splat(simd, lut.scale_factor());

        Self {
            gradient,
            scale_factor,
            lut: lut.lut(),
            t_vals: t_vals.as_chunks::<16>().0.iter(),
            simd,
        }
    }
}

impl<S: Simd> Iterator for GradientPainter<'_, S> {
    type Item = u8x64<S>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let extend = self.gradient.extend;
        let pos = f32x16::from_slice(self.simd, self.t_vals.next()?);
        let t_vals = apply_extend(pos, extend);
        let indices = (t_vals * self.scale_factor).to_int::<u32x16<S>>();

        let mut vals = [0_u8; 64];
        for (val, idx) in vals.as_chunks_mut::<4>().0.iter_mut().zip(*indices) {
            val.copy_from_slice(&self.lut[idx as usize]);
        }

        Some(u8x64::from_slice(self.simd, &vals))
    }
}

impl<S: Simd> crate::fine::Painter for GradientPainter<'_, S> {
    fn paint_u8(mut self, buf: &mut [u8]) {
        self.simd.vectorize(
            #[inline(always)]
            || {
                for chunk in buf.as_chunks_mut::<64>().0 {
                    self.next().unwrap().store_slice(chunk);
                }
            },
        );
    }

    fn paint_f32(self, _: &mut [f32]) {
        unimplemented!()
    }
}
