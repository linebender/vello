// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::fine::common::gradient::SimdGradientKind;
use core::f32::consts::PI;
use vello_common::encode::SweepKind;
use vello_common::fearless_simd::{Simd, SimdBase, SimdFloat, f32x8};

#[derive(Debug)]
pub(crate) struct SimdSweepKind<S: Simd> {
    start_angle: f32x8<S>,
    inv_angle_delta: f32x8<S>,
    simd: S,
}

impl<S: Simd> SimdSweepKind<S> {
    pub(crate) fn new(simd: S, kind: &SweepKind) -> Self {
        Self {
            start_angle: f32x8::splat(simd, kind.start_angle),
            inv_angle_delta: f32x8::splat(simd, kind.inv_angle_delta),
            simd,
        }
    }
}

impl<S: Simd> SimdGradientKind<S> for SimdSweepKind<S> {
    #[inline(always)]
    fn cur_pos(&self, x_pos: f32x8<S>, y_pos: f32x8<S>) -> f32x8<S> {
        let angle = x_y_to_unit_angle(self.simd, x_pos, y_pos) * f32x8::splat(self.simd, 2.0 * PI);

        (angle - self.start_angle) * self.inv_angle_delta
    }
}

#[inline(always)]
fn x_y_to_unit_angle<S: Simd>(simd: S, x: f32x8<S>, y: f32x8<S>) -> f32x8<S> {
    let c0 = f32x8::splat(simd, 0.0);
    let c1 = f32x8::splat(simd, 1.0);
    let c2 = f32x8::splat(simd, 1.0 / 4.0);
    let c3 = f32x8::splat(simd, 1.0 / 2.0);

    let x_abs = x.abs();
    let y_abs = y.abs();

    let slope = x_abs.min(y_abs) / x_abs.max(y_abs);
    let s = slope * slope;

    let a = f32x8::splat(simd, -7.054_738_2e-3).madd(s, f32x8::splat(simd, 2.476_102e-2));
    let b = a.madd(s, f32x8::splat(simd, -5.185_397e-2));
    let c = b.madd(s, f32x8::splat(simd, 0.159_121_17));

    let mut phi = slope * c;

    phi = simd.select_f32x8(simd.simd_lt_f32x8(x_abs, y_abs), c2 - phi, phi);
    phi = simd.select_f32x8(simd.simd_lt_f32x8(x, c0), c3 - phi, phi);
    phi = simd.select_f32x8(simd.simd_lt_f32x8(y, c0), c1 - phi, phi);
    // Clears all NaNs, using the property that NaN != NaN.
    phi = simd.select_f32x8(phi.simd_eq(phi), phi, c0);

    phi
}
