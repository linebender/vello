// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Drawing blurred, rounded rectangles.
//!
//! Implementation is adapted from: <https://git.sr.ht/~raph/blurrr/tree/master/src/distfield.rs>.

use crate::fine::{NumericVec, PosExt, ShaderResultF32};
use crate::kurbo::{Point, Vec2};
use vello_common::encode::EncodedBlurredRoundedRectangle;
use vello_common::fearless_simd::{Simd, SimdBase, SimdFloat, f32x8, u8x16};

#[cfg(not(feature = "std"))]
use vello_common::kurbo::common::FloatFuncs as _;

#[derive(Debug)]
pub(crate) struct BlurredRoundedRectFiller<S: Simd> {
    r: f32x8<S>,
    g: f32x8<S>,
    b: f32x8<S>,
    a: f32x8<S>,
    alpha_calculator: AlphaCalculator<S>,
}

impl<S: Simd> BlurredRoundedRectFiller<S> {
    pub(crate) fn new(
        simd: S,
        rect: &EncodedBlurredRoundedRectangle,
        start_x: u16,
        start_y: u16,
    ) -> Self {
        let start_pos = rect.transform * Point::new(f64::from(start_x), f64::from(start_y));
        let color_components = rect.color.as_premul_f32().components;
        let r = f32x8::splat(simd, color_components[0]);
        let g = f32x8::splat(simd, color_components[1]);
        let b = f32x8::splat(simd, color_components[2]);
        let a = f32x8::splat(simd, color_components[3]);
        let simd_rect = SimdRoundedBlurredRect::new(rect, simd);
        let alpha_calculator =
            AlphaCalculator::new(start_pos, rect.x_advance, rect.y_advance, simd_rect, simd);

        Self {
            alpha_calculator,
            r,
            g,
            b,
            a,
        }
    }
}

impl<S: Simd> Iterator for BlurredRoundedRectFiller<S> {
    type Item = ShaderResultF32<S>;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.alpha_calculator.next().unwrap();
        let r = self.r * next;
        let g = self.g * next;
        let b = self.b * next;
        let a = self.a * next;

        Some(ShaderResultF32 { r, g, b, a })
    }
}

impl<S: Simd> crate::fine::Painter for BlurredRoundedRectFiller<S> {
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

#[derive(Debug)]
struct AlphaCalculator<S: Simd> {
    cur_pos: Point,
    x_advance: Vec2,
    y_advance: Vec2,
    r: SimdRoundedBlurredRect<S>,
    simd: S,
}

impl<S: Simd> AlphaCalculator<S> {
    fn new(
        start_pos: Point,
        x_advance: Vec2,
        y_advance: Vec2,
        r: SimdRoundedBlurredRect<S>,
        simd: S,
    ) -> Self {
        Self {
            cur_pos: start_pos,
            x_advance,
            y_advance,
            r,
            simd,
        }
    }
}

impl<S: Simd> Iterator for AlphaCalculator<S> {
    type Item = f32x8<S>;

    fn next(&mut self) -> Option<Self::Item> {
        let i = f32x8::splat_pos(
            self.simd,
            self.cur_pos.x as f32,
            self.x_advance.x as f32,
            self.y_advance.x as f32,
        );
        let j = f32x8::splat_pos(
            self.simd,
            self.cur_pos.y as f32,
            self.x_advance.y as f32,
            self.y_advance.y as f32,
        );
        let r = &self.r;

        // Equivalent to j + r.v1 - r.v1 * r.height
        let y = j - r.v1.msub(r.height, r.v1);
        // Equivalent to r.r1 + y.abs() - (r.h * r.v1)
        let y0 = r.r1 - r.h.msub(r.v1, y.abs());
        let y1 = y0.max(r.v0);

        // Equivalent to i + r.v1 - r.v1 * r.width
        let x = i - r.v1.msub(r.width, r.v1);
        // Equivalent to r.r1 + x.abs() - (r.w * r.v1)
        let x0 = r.r1 - r.w.msub(r.v1, x.abs());
        let x1 = x0.max(r.v0);
        let d_pos = (x1.powf(r.exponent) + y1.powf(r.exponent)).powf(r.recip_exponent);
        let d_neg = x0.max(y0).min(r.v0);
        let d = d_pos + d_neg - r.r1;
        let z = r.scale
            * (f32x8::compute_erf7(self.simd, r.std_dev_inv * (r.min_edge + d))
                - f32x8::compute_erf7(self.simd, r.std_dev_inv * d));

        self.cur_pos += 2.0 * self.x_advance;

        Some(z)
    }
}

#[derive(Debug)]
struct SimdRoundedBlurredRect<S: Simd> {
    pub exponent: f32,
    pub recip_exponent: f32,
    pub scale: f32x8<S>,
    pub std_dev_inv: f32x8<S>,
    pub min_edge: f32x8<S>,
    pub w: f32x8<S>,
    pub h: f32x8<S>,
    pub width: f32x8<S>,
    pub height: f32x8<S>,
    pub r1: f32x8<S>,
    pub v0: f32x8<S>,
    pub v1: f32x8<S>,
}

impl<S: Simd> SimdRoundedBlurredRect<S> {
    fn new(encoded: &EncodedBlurredRoundedRectangle, s: S) -> Self {
        let h = f32x8::splat(s, encoded.h);
        let w = f32x8::splat(s, encoded.w);
        let width = f32x8::splat(s, encoded.width);
        let height = f32x8::splat(s, encoded.height);
        let r1 = f32x8::splat(s, encoded.r1);
        let exponent = encoded.exponent;
        let recip_exponent = encoded.recip_exponent;
        let scale = f32x8::splat(s, encoded.scale);
        let min_edge = f32x8::splat(s, encoded.min_edge);
        let std_dev_inv = f32x8::splat(s, encoded.std_dev_inv);
        let v0 = f32x8::splat(s, 0.0);
        let v1 = f32x8::splat(s, 0.5);

        Self {
            exponent,
            recip_exponent,
            scale,
            std_dev_inv,
            min_edge,
            w,
            v0,
            v1,
            h,
            width,
            height,
            r1,
        }
    }
}

trait FloatExt<S: Simd> {
    // See https://raphlinus.github.io/audio/2018/09/05/sigmoid.html for a little
    // explanation of this approximation to the erf function.
    // Doing `inline(always)` seems to reduce performance for some reason.
    /// Approximate the erf function.
    fn compute_erf7(simd: S, x: Self) -> Self;
    fn powf(self, x: f32) -> Self;
}

impl<S: Simd> FloatExt<S> for f32x8<S> {
    fn compute_erf7(simd: S, x: Self) -> Self {
        // Clamp `x`, because for large `x` the terms here become `inf`, causing the result to be 0 or
        // `NaN`. This clamping doesn't lose any information, because `erf(±10) ≈ 1` well within `f64`
        // machine precision, let alone `f32`.
        let x = x.max(Self::splat(simd, -10.0)).min(Self::splat(simd, 10.0));
        let x = x * Self::splat(simd, core::f32::consts::FRAC_2_SQRT_PI);
        let xx = x * x;
        let p1 = Self::splat(simd, 0.0104).madd(xx, Self::splat(simd, 0.03395));
        let p2 = p1.madd(xx, Self::splat(simd, 0.24295));
        let p3 = x * xx;
        let x = p2.madd(p3, x);
        let denom = x.madd(x, Self::splat(simd, 1.0)).sqrt();
        x / denom
    }

    #[inline]
    fn powf(mut self, x: f32) -> Self {
        // TODO: SIMD
        self[0] = self[0].powf(x);
        self[1] = self[1].powf(x);
        self[2] = self[2].powf(x);
        self[3] = self[3].powf(x);
        self[4] = self[4].powf(x);
        self[5] = self[5].powf(x);
        self[6] = self[6].powf(x);
        self[7] = self[7].powf(x);

        self
    }
}
