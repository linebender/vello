// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::fine::{NumericVec, PosExt};
use crate::kurbo::Point;
use crate::peniko;
use core::slice::ChunksExact;
use vello_common::encode::{EncodedGradient, GradientLut};
use vello_common::fearless_simd::*;

pub(crate) mod linear;
pub(crate) mod radial;
pub(crate) mod sweep;

pub(crate) fn calculate_t_vals<S: Simd, U: SimdGradientKind<S>>(
    simd: S,
    kind: U,
    buf: &mut [f32],
    gradient: &EncodedGradient,
    start_x: f64,
    start_y: f64,
) {
    simd.vectorize(
        #[inline(always)]
        || {
            let mut cur_pos = gradient.transform * Point::new(start_x, start_y);
            let x_advances = (gradient.x_advance.x as f32, gradient.x_advance.y as f32);
            let y_advances = (gradient.y_advance.x as f32, gradient.y_advance.y as f32);

            for buf_part in buf.chunks_exact_mut(8) {
                let x_pos = f32x8::splat_pos(simd, cur_pos.x as f32, x_advances.0, y_advances.0);
                let y_pos = f32x8::splat_pos(simd, cur_pos.y as f32, x_advances.1, y_advances.1);
                let pos = kind.cur_pos(x_pos, y_pos);
                buf_part.copy_from_slice(pos.as_slice());

                cur_pos += 2.0 * gradient.x_advance;
            }
        },
    );
}

#[derive(Debug)]
pub(crate) struct GradientPainter<'a, S: Simd> {
    gradient: &'a EncodedGradient,
    lut: &'a GradientLut<f32>,
    t_vals: ChunksExact<'a, f32>,
    scale_factor: f32x8<S>,
    simd: S,
}

impl<'a, S: Simd> GradientPainter<'a, S> {
    pub(crate) fn new(simd: S, gradient: &'a EncodedGradient, t_vals: &'a [f32]) -> Self {
        let lut = gradient.f32_lut(simd);
        let scale_factor = f32x8::splat(simd, lut.scale_factor());

        Self {
            gradient,
            scale_factor,
            lut,
            t_vals: t_vals.chunks_exact(8),
            simd,
        }
    }
}

impl<S: Simd> Iterator for GradientPainter<'_, S> {
    type Item = u32x8<S>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let extend = self.gradient.extend;
        let pos = f32x8::from_slice(self.simd, self.t_vals.next()?);
        let t_vals = apply_extend(pos, extend);

        Some((t_vals * self.scale_factor).to_int::<u32x8<S>>())
    }
}

impl<S: Simd> crate::fine::Painter for GradientPainter<'_, S> {
    fn paint_u8(&mut self, buf: &mut [u8]) {
        self.simd.vectorize(
            #[inline(always)]
            || {
                for chunk in buf.chunks_exact_mut(32) {
                    let indices = self.next().unwrap();

                    let rgbas_1: [f32x4<S>; 4] = core::array::from_fn(|i| {
                        f32x4::from_slice(self.simd, &self.lut.get(indices[i] as usize))
                    });
                    let rgbas_1 = self.simd.combine_f32x8(
                        self.simd.combine_f32x4(rgbas_1[0], rgbas_1[1]),
                        self.simd.combine_f32x4(rgbas_1[2], rgbas_1[3]),
                    );
                    let rgbas_1 = u8x16::from_f32(self.simd, rgbas_1);
                    chunk[..16].copy_from_slice(rgbas_1.as_slice());

                    let rgbas_2: [f32x4<S>; 4] = core::array::from_fn(|i| {
                        f32x4::from_slice(self.simd, &self.lut.get(indices[i + 4] as usize))
                    });
                    let rgbas_2 = self.simd.combine_f32x8(
                        self.simd.combine_f32x4(rgbas_2[0], rgbas_2[1]),
                        self.simd.combine_f32x4(rgbas_2[2], rgbas_2[3]),
                    );
                    let rgbas_2 = u8x16::from_f32(self.simd, rgbas_2);
                    chunk[16..].copy_from_slice(rgbas_2.as_slice());
                }
            },
        );
    }

    fn paint_f32(&mut self, buf: &mut [f32]) {
        self.simd.vectorize(
            #[inline(always)]
            || {
                for chunk in buf.chunks_exact_mut(32) {
                    let indices = self.next().unwrap();
                    chunk[0..4].copy_from_slice(&self.lut.get(indices[0] as usize));
                    chunk[4..8].copy_from_slice(&self.lut.get(indices[1] as usize));
                    chunk[8..12].copy_from_slice(&self.lut.get(indices[2] as usize));
                    chunk[12..16].copy_from_slice(&self.lut.get(indices[3] as usize));
                    chunk[16..20].copy_from_slice(&self.lut.get(indices[4] as usize));
                    chunk[20..24].copy_from_slice(&self.lut.get(indices[5] as usize));
                    chunk[24..28].copy_from_slice(&self.lut.get(indices[6] as usize));
                    chunk[28..32].copy_from_slice(&self.lut.get(indices[7] as usize));
                }
            },
        );
    }
}

/// A gradient painter that handles undefined t-values (NaN) by outputting
/// transparent pixels. Used for radial gradient configurations where some
/// pixel positions have no valid t-value (e.g., strip or non-well-behaved
/// focal gradients).
#[derive(Debug)]
pub(crate) struct MaskedGradientPainter<'a, S: Simd> {
    gradient: &'a EncodedGradient,
    lut: &'a GradientLut<f32>,
    t_vals: ChunksExact<'a, f32>,
    scale_factor: f32x8<S>,
    simd: S,
}

impl<'a, S: Simd> MaskedGradientPainter<'a, S> {
    pub(crate) fn new(simd: S, gradient: &'a EncodedGradient, t_vals: &'a [f32]) -> Self {
        let lut = gradient.f32_lut(simd);
        let scale_factor = f32x8::splat(simd, lut.scale_factor());

        Self {
            gradient,
            scale_factor,
            lut,
            t_vals: t_vals.chunks_exact(8),
            simd,
        }
    }

    /// Compute LUT indices and a per-pixel validity mask.
    /// NaN t-values get index 0 (harmless — the sampled value is discarded).
    #[inline(always)]
    fn next_masked(&mut self) -> Option<(u32x8<S>, [i32; 8])> {
        let extend = self.gradient.extend;
        let pos = f32x8::from_slice(self.simd, self.t_vals.next()?);
        let is_defined: [i32; 8] = pos.simd_eq(pos).into();
        let t_vals = apply_extend(pos, extend);
        let indices = (t_vals * self.scale_factor).to_int::<u32x8<S>>();
        let indices = self
            .simd
            .select_u32x8(pos.simd_eq(pos), indices, u32x8::splat(self.simd, 0));
        Some((indices, is_defined))
    }
}

impl<S: Simd> crate::fine::Painter for MaskedGradientPainter<'_, S> {
    fn paint_u8(&mut self, buf: &mut [u8]) {
        self.simd.vectorize(
            #[inline(always)]
            || {
                let zero_f32x4 = f32x4::splat(self.simd, 0.0);
                for chunk in buf.chunks_exact_mut(32) {
                    let (indices, is_defined) = self.next_masked().unwrap();

                    let rgbas_1: [f32x4<S>; 4] = core::array::from_fn(|i| {
                        if is_defined[i] != 0 {
                            f32x4::from_slice(self.simd, &self.lut.get(indices[i] as usize))
                        } else {
                            zero_f32x4
                        }
                    });
                    let rgbas_1 = self.simd.combine_f32x8(
                        self.simd.combine_f32x4(rgbas_1[0], rgbas_1[1]),
                        self.simd.combine_f32x4(rgbas_1[2], rgbas_1[3]),
                    );
                    let rgbas_1 = u8x16::from_f32(self.simd, rgbas_1);
                    chunk[..16].copy_from_slice(rgbas_1.as_slice());

                    let rgbas_2: [f32x4<S>; 4] = core::array::from_fn(|i| {
                        if is_defined[i + 4] != 0 {
                            f32x4::from_slice(self.simd, &self.lut.get(indices[i + 4] as usize))
                        } else {
                            zero_f32x4
                        }
                    });
                    let rgbas_2 = self.simd.combine_f32x8(
                        self.simd.combine_f32x4(rgbas_2[0], rgbas_2[1]),
                        self.simd.combine_f32x4(rgbas_2[2], rgbas_2[3]),
                    );
                    let rgbas_2 = u8x16::from_f32(self.simd, rgbas_2);
                    chunk[16..].copy_from_slice(rgbas_2.as_slice());
                }
            },
        );
    }

    fn paint_f32(&mut self, buf: &mut [f32]) {
        self.simd.vectorize(
            #[inline(always)]
            || {
                let zero = [0.0_f32; 4];
                for chunk in buf.chunks_exact_mut(32) {
                    let (indices, is_defined) = self.next_masked().unwrap();
                    for i in 0..8 {
                        let src = if is_defined[i] != 0 {
                            self.lut.get(indices[i] as usize)
                        } else {
                            zero
                        };
                        chunk[i * 4..(i + 1) * 4].copy_from_slice(&src);
                    }
                }
            },
        );
    }
}

#[inline(always)]
pub(crate) fn apply_extend<S: Simd>(val: f32x8<S>, extend: peniko::Extend) -> f32x8<S> {
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

pub(crate) trait SimdGradientKind<S: Simd> {
    fn cur_pos(&self, x_pos: f32x8<S>, y_pos: f32x8<S>) -> f32x8<S>;
}
