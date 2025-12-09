// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::peniko;
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

impl<S: Simd> Iterator for GradientPainter<'_, S> {
    type Item = u8x64<S>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let extend = self.gradient.extend;
        let pos = f32x16::from_slice(self.simd, self.t_vals.next()?);
        let t_vals = apply_extend(pos, extend);
        let indices = (t_vals * self.scale_factor).to_int::<u32x16<S>>();

        let mut vals = [0_u8; 64];
        for (val, idx) in vals.chunks_exact_mut(4).zip(*indices) {
            val.copy_from_slice(&self.lut[idx as usize]);
        }

        Some(u8x64::from_slice(self.simd, &vals))
    }
}

impl<S: Simd> crate::fine::Painter for GradientPainter<'_, S> {
    fn paint_u8(&mut self, buf: &mut [u8]) {
        self.simd.vectorize(
            #[inline(always)]
            || {
                for chunk in buf.chunks_exact_mut(64) {
                    chunk.copy_from_slice(self.next().unwrap().as_slice());
                }
            },
        );
    }

    fn paint_f32(&mut self, _: &mut [f32]) {
        unimplemented!()
    }
}

// TODO: Maybe delete this method and use `apply_extend` from highp by splitting into two f32x8.
#[inline(always)]
pub(crate) fn apply_extend<S: Simd>(val: f32x16<S>, extend: peniko::Extend) -> f32x16<S> {
    match extend {
        peniko::Extend::Pad => val.max(0.0).min(1.0),
        peniko::Extend::Repeat => (val - val.floor()).fract(),
        // See <https://github.com/google/skia/blob/220738774f7a0ce4a6c7bd17519a336e5e5dea5b/src/opts/SkRasterPipeline_opts.h#L6472-L6475>
        peniko::Extend::Reflect => ((val - 1.0) - 2.0 * ((val - 1.0) * 0.5).floor() - 1.0)
            .abs()
            .max(0.0)
            .min(1.0),
    }
}
