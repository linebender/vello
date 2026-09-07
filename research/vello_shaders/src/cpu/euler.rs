// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

//! Utility functions for Euler Spiral based stroke expansion.

#![expect(
    clippy::excessive_precision,
    reason = "Uses the same constants as the f64 version"
)]

use super::util::Vec2;
use std::f32::consts::FRAC_PI_4;

// Threshold for tangents to be considered near zero length
pub(crate) const TANGENT_THRESH: f32 = 1e-6;

/// This struct contains parameters derived from a cubic Bézier for the
/// purpose of fitting a G1 continuous Euler spiral segment and estimating
/// the Fréchet distance.
///
/// The tangent angles represent deviation from the chord, so that when they
/// are equal, the corresponding Euler spiral is a circular arc.
#[derive(Debug)]
pub(crate) struct CubicParams {
    /// Tangent angle relative to chord at start.
    pub(crate) th0: f32,
    /// Tangent angle relative to chord at end.
    pub(crate) th1: f32,
    /// The effective chord length, always a robustly nonzero value.
    pub(crate) chord_len: f32,
    /// The estimated error between the source cubic and the proposed Euler spiral.
    pub(crate) err: f32,
}

#[derive(Debug)]
pub(crate) struct EulerParams {
    pub(crate) th0: f32,
    // th1 need not be explicitly stored, as it can be derived from k0 - th0
    // See #vello > Euler Spiral `th1` param
    // https://xi.zulipchat.com/#narrow/channel/197075-vello/topic/Euler.20Spiral.20.60th1.60.20param/with/446074149
    pub(crate) k0: f32,
    pub(crate) k1: f32,
    pub(crate) ch: f32,
}

#[derive(Debug)]
pub(crate) struct EulerSeg {
    pub(crate) p0: Vec2,
    pub(crate) p1: Vec2,
    pub(crate) params: EulerParams,
}

impl CubicParams {
    /// Compute parameters from endpoints and derivatives.
    ///
    /// This function is designed to be robust across a wide range of inputs. In
    /// particular, it splits between near-zero chord length and the happy path.
    /// In the former case, the parameters for the Euler spiral would not be valid,
    /// so it proposes a straight line and computes a pretty good (conservative)
    /// estimate of the Fréchet distance between that line and the source cubic.
    ///
    /// Computing an accurate estimate here fixes two tricky cases: very short
    /// lines, in which the error will be below threshold and the flatten logic will
    /// output a single line segment without subdividing, and loop cases with a
    /// short chord, in which case the error will exceed the threshold, and the
    /// chords of the subdivided pieces will be longer.
    ///
    /// An additional case is the near-cusp where the proposed Euler spiral has
    /// a 180 degree U-turn (or, more generally, one angle exceeds 90 degrees and
    /// the other does not). In that case, the resulting Euler spiral is quite
    /// well defined (with finite curvature, so that its offset will generate a
    /// near-semicircle, preserving G1 continuity), but the analytic error
    /// calculation would be a huge overestimate. In that case, we just return
    /// a rough estimate of the distance between the chord and the spiral segment.
    pub(crate) fn from_points_derivs(p0: Vec2, p1: Vec2, q0: Vec2, q1: Vec2, dt: f32) -> Self {
        let chord = p1 - p0;
        let chord_squared = chord.length_squared();
        let chord_len = chord_squared.sqrt();
        // Chord is near-zero; straight line case.
        if chord_squared < TANGENT_THRESH.powi(2) {
            // This error estimate was determined empirically through randomized
            // testing, though it is likely it can be derived analytically.
            let chord_err = ((9. / 32.0) * (q0.length_squared() + q1.length_squared())).sqrt() * dt;
            return Self {
                th0: 0.0,
                th1: 0.0,
                chord_len: TANGENT_THRESH,
                err: chord_err,
            };
        }
        let scale = dt / chord_squared;
        let h0 = Vec2::new(
            q0.x * chord.x + q0.y * chord.y,
            q0.y * chord.x - q0.x * chord.y,
        );
        let th0 = h0.atan2();
        let d0 = h0.length() * scale;
        let h1 = Vec2::new(
            q1.x * chord.x + q1.y * chord.y,
            q1.x * chord.y - q1.y * chord.x,
        );
        let th1 = h1.atan2();
        let d1 = h1.length() * scale;
        // Robustness note: we may want to clamp the magnitude of the angles to
        // a bit less than pi. Perhaps here, perhaps downstream.

        // Estimate error of geometric Hermite interpolation to Euler spiral.
        let cth0 = th0.cos();
        let cth1 = th1.cos();
        let mut err = if cth0 * cth1 < 0.0 {
            // A value of 2.0 represents the approximate worst case distance
            // from an Euler spiral with 0 and pi tangents to the chord. It
            // is not very critical; doubling the value would result in one more
            // subdivision in effectively a binary search for the cusp, while too
            // small a value may result in the actual error exceeding the bound.
            2.0
        } else {
            // Protect against divide-by-zero. This happens with a double cusp, so
            // should in the general case cause subdivisions.
            let e0 = (2. / 3.) / (1.0 + cth0).max(1e-9);
            let e1 = (2. / 3.) / (1.0 + cth1).max(1e-9);
            let s0 = th0.sin();
            let s1 = th1.sin();
            // Note: some other versions take sin of s0 + s1 instead. Those are incorrect.
            // Strangely, calibration is the same, but more work could be done.
            let s01 = cth0 * s1 + cth1 * s0;
            let amin = 0.15 * (2. * e0 * s0 + 2. * e1 * s1 - e0 * e1 * s01);
            let a = 0.15 * (2. * d0 * s0 + 2. * d1 * s1 - d0 * d1 * s01);
            let aerr = (a - amin).abs();
            let symm = (th0 + th1).abs();
            let asymm = (th0 - th1).abs();
            let dist = (d0 - e0).hypot(d1 - e1);
            let ctr = 4.625e-6 * symm.powi(5) + 7.5e-3 * asymm * symm.powi(2);
            let halo_symm = 5e-3 * symm * dist;
            let halo_asymm = 7e-2 * asymm * dist;
            /*
            println!("    e0: {e0}");
            println!("    e1: {e1}");
            println!("    s0: {s0}");
            println!("    s1: {s1}");
            println!("    s01: {s01}");
            println!("    amin: {amin}");
            println!("    a: {a}");
            println!("    aerr: {aerr}");
            println!("    symm: {symm}");
            println!("    asymm: {asymm}");
            println!("    dist: {dist}");
            println!("    ctr: {ctr}");
            println!("    halo_symm: {halo_symm}");
            println!("    halo_asymm: {halo_asymm}");
            */
            ctr + 1.55 * aerr + halo_symm + halo_asymm
        };
        err *= chord_len;
        Self {
            th0,
            th1,
            chord_len,
            err,
        }
    }
}

impl EulerParams {
    pub(crate) fn from_angles(th0: f32, th1: f32) -> Self {
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
        Self { th0, k0, k1, ch }
    }

    pub(crate) fn eval_th(&self, t: f32) -> f32 {
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

    // Offset provided is in same units as curve; chord is normalized to (1, 0).
    fn eval_with_offset(&self, t: f32, offset: f32) -> Vec2 {
        let th = self.eval_th(t);
        let v = Vec2::new(offset * th.sin(), offset * th.cos());
        self.eval(t) + v
    }
}

impl EulerSeg {
    pub(crate) fn from_params(p0: Vec2, p1: Vec2, params: EulerParams) -> Self {
        Self { p0, p1, params }
    }

    #[expect(unused, reason = "Unclear why this code exists")]
    pub(crate) fn eval(&self, t: f32) -> Vec2 {
        let Vec2 { x, y } = self.params.eval(t);
        let chord = self.p1 - self.p0;
        Vec2::new(
            self.p0.x + chord.x * x - chord.y * y,
            self.p0.y + chord.x * y + chord.y * x,
        )
    }

    // Note: offset provided is normalized so that 1 = chord length, while
    // the return value is in the same coordinate space as the endpoints.
    pub(crate) fn eval_with_offset(&self, t: f32, normalized_offset: f32) -> Vec2 {
        let chord = self.p1 - self.p0;
        let Vec2 { x, y } = self.params.eval_with_offset(t, normalized_offset);
        Vec2::new(
            self.p0.x + chord.x * x - chord.y * y,
            self.p0.y + chord.x * y + chord.y * x,
        )
    }
}

/// Integrate Euler spiral.
///
/// This is a 10th order polynomial. The evaluation method is explained in
/// Raph's thesis in section 8.1.2.
///
/// This choice of polynomial is fairly conservative, as it will produce
/// very good accuracy for angles up to about 1 radian, and those angles
/// should almost never happen (the exception being cusps). One possibility
/// to explore is using a lower degree polynomial here, but changing the
/// tuning parameters for subdivision so the additional error here is also
/// factored into the error threshold. However, two cautions against that:
/// First, it doesn't really address the cusp case, where angles will remain
/// large even after further subdivision, and second, the cost of even this
/// more conservative choice is modest; it's just some multiply-adds.
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

pub(crate) fn espc_int_approx(x: f32) -> f32 {
    let y = x.abs();
    let a = if y < BREAK1 {
        (SIN_SCALE * y).sin() * (1.0 / SIN_SCALE)
    } else if y < BREAK2 {
        (8.0_f32.sqrt() / 3.0) * (y - 1.0) * (y - 1.0).abs().sqrt() + FRAC_PI_4
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

pub(crate) fn espc_int_inv_approx(x: f32) -> f32 {
    let y = x.abs();
    let a = if y < 0.7010707591262915 {
        (x * SIN_SCALE).asin() * (1.0 / SIN_SCALE)
    } else if y < 0.903249293595206 {
        let b = y - FRAC_PI_4;
        let u = b.abs().powf(2. / 3.).copysign(b);
        u * (9.0_f32 / 8.).cbrt() + 1.0
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
