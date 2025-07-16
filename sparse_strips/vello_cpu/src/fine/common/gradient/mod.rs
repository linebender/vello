// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::fine::{NumericVec, PosExt, ShaderResultF32};
use crate::kurbo::Point;
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
    start_x: u16,
    start_y: u16,
) {
    let mut cur_pos = gradient.transform * Point::new(f64::from(start_x), f64::from(start_y));
    let x_advances = (gradient.x_advance.x as f32, gradient.x_advance.y as f32);
    let y_advances = (gradient.y_advance.x as f32, gradient.y_advance.y as f32);

    for buf_part in buf.chunks_exact_mut(8) {
        let x_pos = f32x8::splat_pos(simd, cur_pos.x as f32, x_advances.0, y_advances.0);
        let y_pos = f32x8::splat_pos(simd, cur_pos.y as f32, x_advances.1, y_advances.1);
        let pos = kind.cur_pos(x_pos, y_pos);
        buf_part.copy_from_slice(&pos.val);

        cur_pos += 2.0 * gradient.x_advance;
    }
}

#[derive(Debug)]
pub(crate) struct GradientPainter<'a, S: Simd> {
    gradient: &'a EncodedGradient,
    lut: &'a GradientLut<f32>,
    t_vals: ChunksExact<'a, f32>,
    has_undefined: bool,
    scale_factor: f32x8<S>,
    simd: S,
}

impl<'a, S: Simd> GradientPainter<'a, S> {
    pub(crate) fn new(
        simd: S,
        gradient: &'a EncodedGradient,
        has_undefined: bool,
        t_vals: &'a [f32],
    ) -> Self {
        let lut = gradient.f32_lut(simd);
        let scale_factor = f32x8::splat(simd, lut.scale_factor());

        Self {
            gradient,
            scale_factor,
            has_undefined,
            lut,
            t_vals: t_vals.chunks_exact(8),
            simd,
        }
    }
}

impl<'a, S: Simd> Iterator for GradientPainter<'a, S> {
    type Item = ShaderResultF32<S>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let pad = self.gradient.pad;
        let pos = f32x8::from_slice(self.simd, self.t_vals.next()?);
        let t_vals = extend(pos, pad);
        let indices = (t_vals * self.scale_factor).cvt_u32();

        let mut r = [0.0_f32; 8];
        let mut g = [0.0_f32; 8];
        let mut b = [0.0_f32; 8];
        let mut a = [0.0_f32; 8];

        // TODO: Investigate whether we can use a loop without performance hit.
        macro_rules! gather {
            ($idx:expr) => {
                let sample = self.lut.get(indices[$idx] as usize);
                r[$idx] = sample[0];
                g[$idx] = sample[1];
                b[$idx] = sample[2];
                a[$idx] = sample[3];
            };
        }

        gather!(0);
        gather!(1);
        gather!(2);
        gather!(3);
        gather!(4);
        gather!(5);
        gather!(6);
        gather!(7);

        let mut r = f32x8::from_slice(self.simd, &r);
        let mut g = f32x8::from_slice(self.simd, &g);
        let mut b = f32x8::from_slice(self.simd, &b);
        let mut a = f32x8::from_slice(self.simd, &a);

        if self.has_undefined {
            macro_rules! mask_nan {
                ($channel:expr) => {
                    $channel = self.simd.select_f32x8(
                        // On some architectures, the NaNs of `t_vals` might have been cleared already by
                        // the `extend` function, so use the original variable as the mask.
                        // Mask out NaNs with 0.
                        self.simd.simd_eq_f32x8(pos, pos),
                        $channel,
                        f32x8::splat(self.simd, 0.0),
                    );
                };
            }

            mask_nan!(r);
            mask_nan!(g);
            mask_nan!(b);
            mask_nan!(a);
        }

        Some(ShaderResultF32 { r, g, b, a })
    }
}

impl<S: Simd> crate::fine::Painter for GradientPainter<'_, S> {
    fn paint_u8(&mut self, buf: &mut [u8]) {
        for chunk in buf.chunks_exact_mut(64) {
            let first = self.next().unwrap();
            let simd = first.r.simd;
            let second = self.next().unwrap();

            let r = u8x16::from_f32(simd, simd.combine_f32x8(first.r, second.r));
            let g = u8x16::from_f32(simd, simd.combine_f32x8(first.g, second.g));
            let b = u8x16::from_f32(simd, simd.combine_f32x8(first.b, second.b));
            let a = u8x16::from_f32(simd, simd.combine_f32x8(first.a, second.a));

            let combined = simd.combine_u8x32(simd.combine_u8x16(r, g), simd.combine_u8x16(b, a));

            simd.store_interleaved_128_u8x64(combined, (&mut chunk[..]).try_into().unwrap());
        }
    }

    fn paint_f32(&mut self, buf: &mut [f32]) {
        for chunk in buf.chunks_exact_mut(32) {
            let (c1, c2) = self.next().unwrap().get();
            c1.simd
                .store_interleaved_128_f32x16(c1, (&mut chunk[..16]).try_into().unwrap());
            c2.simd
                .store_interleaved_128_f32x16(c2, (&mut chunk[16..]).try_into().unwrap());
        }
    }
}

#[inline(always)]
pub(crate) fn extend<S: Simd>(val: f32x8<S>, pad: bool) -> f32x8<S> {
    if pad {
        val.max(0.0).min(1.0)
    } else {
        (val - val.floor()).fract()
    }
}

pub(crate) trait SimdGradientKind<S: Simd> {
    fn cur_pos(&self, x_pos: f32x8<S>, y_pos: f32x8<S>) -> f32x8<S>;
}
