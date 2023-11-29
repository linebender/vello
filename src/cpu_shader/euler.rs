// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

//! Utility functions for Euler-Spiral-based  

use std::f32::consts::FRAC_PI_4;
use super::util::Vec2;

#[derive(Debug)]
pub struct CubicParams {
    pub th0: f32,
    pub th1: f32,
    pub d0: f32,
    pub d1: f32,
}

#[derive(Debug)]
pub struct EulerParams {
    pub th0: f32,
    pub th1: f32,
    pub k0: f32,
    pub k1: f32,
    pub ch: f32,
}

#[derive(Debug)]
pub struct EulerSeg {
    pub p0: Vec2,
    pub p1: Vec2,
    pub params: EulerParams,
}

impl CubicParams {
    pub fn from_cubic(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2) -> Self {
        let chord = p3 - p0;
        // TODO: if chord is 0, we have a problem
        let d01 = p1 - p0;
        let h0 = Vec2::new(
            d01.x * chord.x + d01.y * chord.y,
            d01.y * chord.x - d01.x * chord.y,
        );
        let th0 = h0.atan2();
        let d0 = h0.length() / chord.length_squared();
        let d23 = p3 - p2;
        let h1 = Vec2::new(
            d23.x * chord.x + d23.y * chord.y,
            d23.x * chord.y - d23.y * chord.x,
        );
        let th1 = h1.atan2();
        let d1 = h1.length() / chord.length_squared();
        CubicParams { th0, th1, d0, d1 }
    }

    // Estimated error of GH to Euler spiral
    //
    // Return value is normalized to chord - to get actual error, multiply
    // by chord.
    pub fn est_euler_err(&self) -> f32 {
        // Potential optimization: work with unit vector rather than angle
        let e0 = (2. / 3.) / (1.0 + self.th0.cos());
        let e1 = (2. / 3.) / (1.0 + self.th1.cos());
        let s0 = self.th0.sin();
        let s1 = self.th1.sin();
        let s01 = (s0 + s1).sin();
        let amin = 0.15 * (2. * e0 * s0 + 2. * e1 * s1 - e0 * e1 * s01);
        let a = 0.15 * (2. * self.d0 * s0 + 2. * self.d1 * s1 - self.d0 * self.d1 * s01);
        let aerr = (a - amin).abs();
        let symm = (self.th0 + self.th1).abs();
        let asymm = (self.th0 - self.th1).abs();
        let dist = (self.d0 - e0).hypot(self.d1 - e1);
        let ctr = 3.7e-6 * symm.powi(5) + 6e-3 * asymm * symm.powi(2);
        let halo_symm = 5e-3 * symm * dist;
        let halo_asymm = 7e-2 * asymm * dist;
        1.25 * ctr + 1.55 * aerr + halo_symm + halo_asymm
    }
}

impl EulerParams {
    pub fn from_angles(th0: f32, th1: f32) -> EulerParams {
        let k0 = th0 + th1;
        let dth = th1 - th0;
        let d2 = dth * dth;
        let k2 = k0 * k0;
        let mut a = 6.0;
        a -= d2 * (1. / 70.);
        a -= (d2 * d2) * (1. / 10780.);
        a += (d2 * d2 * d2) * 2.769178184818219e-07;
        let b = -0.1 + d2 * (1. / 4200.) + d2 * d2 * 1.6959677820260655e-05;
        let c = -1. / 1400. + d2 * 6.84915970574303e-05 - k2 * 7.936475029053326e-06;
        a += (b + c * k2) * k2;
        let k1 = dth * a;

        // calculation of chord
        let mut ch = 1.0;
        ch -= d2 * (1. / 40.);
        ch += (d2 * d2) * 0.00034226190482569864;
        ch -= (d2 * d2 * d2) * 1.9349474568904524e-06;
        let b = -1. / 24. + d2 * 0.0024702380951963226 - d2 * d2 * 3.7297408997537985e-05;
        let c = 1. / 1920. - d2 * 4.87350869747975e-05 - k2 * 3.1001936068463107e-06;
        ch += (b + c * k2) * k2;
        EulerParams {
            th0,
            th1,
            k0,
            k1,
            ch,
        }
    }

    pub fn eval_th(&self, t: f32) -> f32 {
        (self.k0 + 0.5 * self.k1 * (t - 1.0)) * t - self.th0
    }

    /// Evaluate the curve at the given parameter.
    ///
    /// The parameter is in the range 0..1, and the result goes from (0, 0) to (1, 0).
    fn eval(&self, t: f32) -> Vec2 {
        let thm = self.eval_th(t * 0.5);
        let k0 = self.k0;
        let k1 = self.k1;
        let (u, v) = integ_euler_10((k0 + k1 * (0.5 * t - 0.5)) * t, k1 * t * t);
        let s = t / self.ch * thm.sin();
        let c = t / self.ch * thm.cos();
        let x = u * c - v * s;
        let y = -v * c - u * s;
        Vec2::new(x, y)
    }

    fn eval_with_offset(&self, t: f32, offset: f32) -> Vec2 {
        let th = self.eval_th(t);
        let v = Vec2::new(offset * th.sin(), offset * th.cos());
        self.eval(t) + v
    }
}

impl EulerSeg {
    pub fn from_params(p0: Vec2, p1: Vec2, params: EulerParams) -> Self {
        EulerSeg { p0, p1, params }
    }

    #[allow(unused)]
    pub fn eval(&self, t: f32) -> Vec2 {
        let Vec2 { x, y } = self.params.eval(t);
        let chord = self.p1 - self.p0;
        Vec2::new(
            self.p0.x + chord.x * x - chord.y * y,
            self.p0.y + chord.x * y + chord.y * x,
        )
    }

    pub fn eval_with_offset(&self, t: f32, offset: f32) -> Vec2 {
        let chord = self.p1 - self.p0;
        let scaled = offset / chord.length();
        let Vec2 { x, y } = self.params.eval_with_offset(t, scaled);
        Vec2::new(
            self.p0.x + chord.x * x - chord.y * y,
            self.p0.y + chord.x * y + chord.y * x,
        )
    }
}

/// Integrate Euler spiral.
///
/// TODO: investigate needed accuracy. We might be able to get away
/// with 8th order.
fn integ_euler_10(k0: f32, k1: f32) -> (f32, f32) {
    let t1_1 = k0;
    let t1_2 = 0.5 * k1;
    let t2_2 = t1_1 * t1_1;
    let t2_3 = 2. * (t1_1 * t1_2);
    let t2_4 = t1_2 * t1_2;
    let t3_4 = t2_2 * t1_2 + t2_3 * t1_1;
    let t3_6 = t2_4 * t1_2;
    let t4_4 = t2_2 * t2_2;
    let t4_5 = 2. * (t2_2 * t2_3);
    let t4_6 = 2. * (t2_2 * t2_4) + t2_3 * t2_3;
    let t4_7 = 2. * (t2_3 * t2_4);
    let t4_8 = t2_4 * t2_4;
    let t5_6 = t4_4 * t1_2 + t4_5 * t1_1;
    let t5_8 = t4_6 * t1_2 + t4_7 * t1_1;
    let t6_6 = t4_4 * t2_2;
    let t6_7 = t4_4 * t2_3 + t4_5 * t2_2;
    let t6_8 = t4_4 * t2_4 + t4_5 * t2_3 + t4_6 * t2_2;
    let t7_8 = t6_6 * t1_2 + t6_7 * t1_1;
    let t8_8 = t6_6 * t2_2;
    let mut u = 1.;
    u -= (1. / 24.) * t2_2 + (1. / 160.) * t2_4;
    u += (1. / 1920.) * t4_4 + (1. / 10752.) * t4_6 + (1. / 55296.) * t4_8;
    u -= (1. / 322560.) * t6_6 + (1. / 1658880.) * t6_8;
    u += (1. / 92897280.) * t8_8;
    let mut v = (1. / 12.) * t1_2;
    v -= (1. / 480.) * t3_4 + (1. / 2688.) * t3_6;
    v += (1. / 53760.) * t5_6 + (1. / 276480.) * t5_8;
    v -= (1. / 11612160.) * t7_8;
    (u, v)
}

const BREAK1: f32 = 0.8;
const BREAK2: f32 = 1.25;
const BREAK3: f32 = 2.1;
const SIN_SCALE: f32 = 1.0976991822760038;
const QUAD_A1: f32 = 0.6406;
const QUAD_B1: f32 = -0.81;
const QUAD_C1: f32 = 0.9148117935952064;
const QUAD_A2: f32 = 0.5;
const QUAD_B2: f32 = -0.156;
const QUAD_C2: f32 = 0.16145779359520596;

pub fn espc_int_approx(x: f32) -> f32 {
    let y = x.abs();
    let a = if y < BREAK1 {
        (SIN_SCALE * y).sin() * (1.0 / SIN_SCALE)
    } else if y < BREAK2 {
        (8.0f32.sqrt() / 3.0) * (y - 1.0) * (y - 1.0).abs().sqrt() + FRAC_PI_4
    } else {
        let (a, b, c) = if y < BREAK3 {
            (QUAD_A1, QUAD_B1, QUAD_C1)
        } else {
            (QUAD_A2, QUAD_B2, QUAD_C2)
        };
        a * y * y + b * y + c
    };
    a.copysign(x)
}

pub fn espc_int_inv_approx(x: f32) -> f32 {
    let y = x.abs();
    let a = if y < 0.7010707591262915 {
        (x * SIN_SCALE).asin() * (1.0 / SIN_SCALE)
    } else if y < 0.903249293595206 {
        let b = y - FRAC_PI_4;
        let u = b.abs().powf(2. / 3.).copysign(b);
        u * (9.0f32 / 8.).cbrt() + 1.0
    } else {
        let (u, v, w) = if y < 2.038857793595206 {
            const B: f32 = 0.5 * QUAD_B1 / QUAD_A1;
            (B * B - QUAD_C1 / QUAD_A1, 1.0 / QUAD_A1, B)
        } else {
            const B: f32 = 0.5 * QUAD_B2 / QUAD_A2;
            (B * B - QUAD_C2 / QUAD_A2, 1.0 / QUAD_A2, B)
        };
        (u + v * y).sqrt() - w
    };
    a.copysign(x)
}
