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

const GRADIENT_INVALID_POS: u32 = u32::MAX;

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
                pos.store_slice(buf_part);

                cur_pos += 2.0 * gradient.x_advance;
            }
        },
    );
}

#[derive(Debug, Clone)]
pub(crate) struct GradientPainter<'a, S: Simd> {
    gradient: &'a EncodedGradient,
    lut: &'a GradientLut<f32>,
    t_vals: ChunksExact<'a, f32>,
    has_undefined: bool,
    scale_factor: f32x8<S>,
    simd: S,
}

impl<'a, S: Simd> GradientPainter<'a, S> {
    pub(crate) fn new(simd: S, gradient: &'a EncodedGradient, t_vals: &'a [f32]) -> Self {
        let lut = gradient.f32_lut(simd);
        let scale_factor: f32x8<S> = f32x8::splat(simd, lut.scale_factor());

        Self {
            gradient,
            scale_factor,
            lut,
            t_vals: t_vals.chunks_exact(8),
            has_undefined: gradient.has_undefined,
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
        let valid = pos.simd_eq(pos);
        let indices = (t_vals * self.scale_factor).to_int::<u32x8<S>>();
        // In case we had any NaN's, set the index to an explicit invalid sentinel. There
        // probably is architecture-specific behavior how NaN will be converted to
        // f32, so we apply our own handling of that.
        let indices = self.simd.select_u32x8(
            valid,
            indices,
            u32x8::splat(self.simd, GRADIENT_INVALID_POS),
        );

        Some(indices)
    }
}

impl<S: Simd> crate::fine::Painter for GradientPainter<'_, S> {
    fn paint_u8(&mut self, buf: &mut [u8]) {
        let max_index = self.lut.width() as u32 - 1;
        let mut masked_iter = self.has_undefined.then_some(self.clone());

        self.simd.vectorize(
            #[inline(always)]
            || {
                let max_index = u32x8::splat(self.simd, max_index);

                for chunk in buf.chunks_exact_mut(32) {
                    let indices = self.next().unwrap();
                    let clamped_indices = indices.min(max_index);

                    let rgbas_1: [f32x4<S>; 4] = core::array::from_fn(|i| {
                        let idx = clamped_indices[i] as usize;
                        f32x4::from_slice(self.simd, &self.lut.get(idx))
                    });
                    let rgbas_1 = self.simd.combine_f32x8(
                        self.simd.combine_f32x4(rgbas_1[0], rgbas_1[1]),
                        self.simd.combine_f32x4(rgbas_1[2], rgbas_1[3]),
                    );
                    let rgbas_1 = u8x16::from_f32(self.simd, rgbas_1);
                    rgbas_1.store_slice(&mut chunk[..16]);

                    let rgbas_2: [f32x4<S>; 4] = core::array::from_fn(|i| {
                        let idx = clamped_indices[i + 4] as usize;
                        f32x4::from_slice(self.simd, &self.lut.get(idx))
                    });
                    let rgbas_2 = self.simd.combine_f32x8(
                        self.simd.combine_f32x4(rgbas_2[0], rgbas_2[1]),
                        self.simd.combine_f32x4(rgbas_2[2], rgbas_2[3]),
                    );
                    let rgbas_2 = u8x16::from_f32(self.simd, rgbas_2);
                    rgbas_2.store_slice(&mut chunk[16..]);
                }
            },
        );

        // Mask any potential NaN positions.
        if let Some(mut masked_iter) = masked_iter.take() {
            self.simd.vectorize(
                #[inline(always)]
                || {
                    for chunk in buf.chunks_exact_mut(32) {
                        let indices = masked_iter.next().unwrap();
                        let invalid =
                            indices.simd_eq(u32x8::splat(self.simd, GRADIENT_INVALID_POS));
                        let (invalid_1, invalid_2) = self.simd.split_mask32x8(invalid);

                        let loaded_1 = self
                            .simd
                            .reinterpret_u32_u8x16(u8x16::from_slice(self.simd, &chunk[..16]));
                        let masked_1 =
                            self.simd
                                .select_u32x4(invalid_1, u32x4::splat(self.simd, 0), loaded_1);
                        self.simd
                            .reinterpret_u8_u32x4(masked_1)
                            .store_slice(&mut chunk[..16]);

                        let loaded_2 = self
                            .simd
                            .reinterpret_u32_u8x16(u8x16::from_slice(self.simd, &chunk[16..]));
                        let masked_2 =
                            self.simd
                                .select_u32x4(invalid_2, u32x4::splat(self.simd, 0), loaded_2);
                        self.simd
                            .reinterpret_u8_u32x4(masked_2)
                            .store_slice(&mut chunk[16..]);
                    }
                },
            );
        }
    }

    fn paint_f32(&mut self, buf: &mut [f32]) {
        let mut masked_iter = self.has_undefined.then_some(self.clone());

        self.simd.vectorize(
            #[inline(always)]
            || {
                let max_index = self.lut.width() as u32 - 1;
                let max_index = u32x8::splat(self.simd, max_index);

                for chunk in buf.chunks_exact_mut(32) {
                    let indices = self.next().unwrap();
                    let clamped_indices = indices.min(max_index);
                    chunk[0..4].copy_from_slice(&self.lut.get(clamped_indices[0] as usize));
                    chunk[4..8].copy_from_slice(&self.lut.get(clamped_indices[1] as usize));
                    chunk[8..12].copy_from_slice(&self.lut.get(clamped_indices[2] as usize));
                    chunk[12..16].copy_from_slice(&self.lut.get(clamped_indices[3] as usize));
                    chunk[16..20].copy_from_slice(&self.lut.get(clamped_indices[4] as usize));
                    chunk[20..24].copy_from_slice(&self.lut.get(clamped_indices[5] as usize));
                    chunk[24..28].copy_from_slice(&self.lut.get(clamped_indices[6] as usize));
                    chunk[28..32].copy_from_slice(&self.lut.get(clamped_indices[7] as usize));
                }
            },
        );

        // Mask any potential NaN positions.
        if let Some(mut masked_iter) = masked_iter.take() {
            self.simd.vectorize(
                #[inline(always)]
                || {
                    for chunk in buf.chunks_exact_mut(32) {
                        let indices = masked_iter.next().unwrap();
                        let (indices_1, indices_2) = self.simd.split_u32x8(indices);
                        let invalid_1 = invalid_f32_mask(self.simd, indices_1);
                        let loaded_1 = f32x16::from_slice(self.simd, &chunk[..16]);
                        let masked_1 = self.simd.select_f32x16(
                            invalid_1,
                            f32x16::splat(self.simd, 0.0),
                            loaded_1,
                        );
                        masked_1.store_slice(&mut chunk[..16]);

                        let invalid_2 = invalid_f32_mask(self.simd, indices_2);
                        let loaded_2 = f32x16::from_slice(self.simd, &chunk[16..]);
                        let masked_2 = self.simd.select_f32x16(
                            invalid_2,
                            f32x16::splat(self.simd, 0.0),
                            loaded_2,
                        );
                        masked_2.store_slice(&mut chunk[16..]);
                    }
                },
            );
        }
    }
}

#[inline(always)]
fn invalid_f32_mask<S: Simd>(simd: S, indices: u32x4<S>) -> mask32x16<S> {
    let indices = indices.zip_low(indices).combine(indices.zip_high(indices));
    let indices = indices.zip_low(indices).combine(indices.zip_high(indices));
    indices.simd_eq(u32x16::splat(simd, GRADIENT_INVALID_POS))
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
