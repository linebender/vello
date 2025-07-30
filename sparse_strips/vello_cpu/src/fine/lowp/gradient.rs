// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use core::slice::ChunksExact;
use vello_common::encode::EncodedGradient;
use vello_common::fearless_simd::*;

/// An accelerated gradient painter for u8.
///
/// Assumes that the gradient has no undefined positions.
#[derive(Debug)]
pub(crate) struct GradientPainter<'a, S: Simd> {
    gradient: &'a EncodedGradient,
    lut: &'a [[u8; 4]],
    t_vals: ChunksExact<'a, f32>,
    scale_factor: f32x16<S>,
    simd: S,
}

impl<'a, S: Simd> GradientPainter<'a, S> {
    pub(crate) fn new(simd: S, gradient: &'a EncodedGradient, t_vals: &'a [f32]) -> Self {
        let lut = gradient.u8_lut(simd);
        let scale_factor = f32x16::splat(simd, lut.scale_factor());

        Self {
            gradient,
            scale_factor,
            lut: lut.lut(),
            t_vals: t_vals.chunks_exact(16),
            simd,
        }
    }
}

impl<'a, S: Simd> Iterator for GradientPainter<'a, S> {
    type Item = u8x64<S>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let pad = self.gradient.pad;
        let pos = f32x16::from_slice(self.simd, self.t_vals.next()?);
        let t_vals = extend(pos, pad);
        let indices = (t_vals * self.scale_factor).cvt_u32();

        let mut vals = [0_u8; 64];
        for (val, idx) in vals.chunks_exact_mut(4).zip(indices.val) {
            val.copy_from_slice(&self.lut[idx as usize]);
        }

        Some(u8x64::from_slice(self.simd, &vals))
    }
}

impl<S: Simd> crate::fine::Painter for GradientPainter<'_, S> {
    fn paint_u8(&mut self, buf: &mut [u8]) {
        for chunk in buf.chunks_exact_mut(64) {
            chunk.copy_from_slice(&self.next().unwrap().val);
        }
    }

    fn paint_f32(&mut self, _: &mut [f32]) {
        unimplemented!()
    }
}

#[inline(always)]
pub(crate) fn extend<S: Simd>(val: f32x16<S>, pad: bool) -> f32x16<S> {
    if pad {
        val.max(0.0).min(1.0)
    } else {
        (val - val.floor()).fract()
    }
}
