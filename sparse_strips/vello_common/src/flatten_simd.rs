// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This is a temporary module that contains a SIMD version of flattening of cubic curves, as
//! well as some code that was copied from kurbo, which is needed to reimplement the
//! full `flatten` method.

use crate::flatten::{SQRT_TOL, TOL, TOL_2};
#[cfg(not(feature = "std"))]
use crate::kurbo::common::FloatFuncs as _;
use crate::kurbo::{CubicBez, Line, ParamCurve, ParamCurveNearest, PathEl, Point, QuadBez};
use alloc::vec::Vec;
use bytemuck::{Pod, Zeroable};
use fearless_simd::*;

/// The element of a path made of lines.
///
/// Each subpath must start with a `MoveTo`. Closing of subpaths is not supported, and subpaths are
/// not closed implicitly when a new subpath (with `MoveTo`) is started. It is expected that closed
/// subpaths are watertight in the sense that the last `LineTo` matches exactly with the first
/// `MoveTo`.
///
/// This intentionally allows for non-watertight subpaths, as, e.g., lines that are fully outside
/// of the viewport do not need to be drawn.
///
/// See [`PathEl`] for a more general-purpose path element type.
pub(crate) enum LinePathEl {
    MoveTo(Point),
    LineTo(Point),
}

// Unlike kurbo, which takes a closure with a callback for outputting the lines, we use a trait
// instead. The reason is that this way the callback can be inlined, which is not possible with
// a closure and turned out to have a noticeable overhead.
pub(crate) trait Callback {
    fn callback(&mut self, el: LinePathEl);
}

/// See the docs for the kurbo implementation of flattening:
/// <https://docs.rs/kurbo/latest/kurbo/fn.flatten.html>
///
/// This version works using a similar approach but using f32x4/f32x8 SIMD instead.
#[inline(always)]
pub(crate) fn flatten<S: Simd>(
    simd: S,
    path: impl IntoIterator<Item = PathEl>,
    callback: &mut impl Callback,
    flatten_ctx: &mut FlattenCtx,
) {
    flatten_ctx.flattened_cubics.clear();

    let mut closed = true;
    let mut start_pt = Point::ZERO;
    let mut last_pt = Point::ZERO;

    for el in path {
        match el {
            PathEl::MoveTo(p) => {
                if !closed && last_pt != start_pt {
                    callback.callback(LinePathEl::LineTo(start_pt));
                }
                closed = false;
                last_pt = p;
                start_pt = p;
                callback.callback(LinePathEl::MoveTo(p));
            }
            PathEl::LineTo(p) => {
                debug_assert!(!closed, "Expected a `MoveTo` before a `LineTo`");
                last_pt = p;
                callback.callback(LinePathEl::LineTo(p));
            }
            PathEl::QuadTo(p1, p2) => {
                debug_assert!(!closed, "Expected a `MoveTo` before a `QuadTo`");
                let p0 = last_pt;
                // An upper bound on the shortest distance of any point on the quadratic Bezier
                // curve to the line segment [p0, p2] is 1/2 of the control-point-to-line-segment
                // distance.
                //
                // The derivation is similar to that for the cubic Bezier (see below). In
                // short:
                //
                // q(t) = B0(t) p0 + B1(t) p1 + B2(t) p2
                // dist(q(t), [p0, p1]) <= B1(t) dist(p1, [p0, p1])
                //                       = 2 (1-t)t dist(p1, [p0, p1]).
                //
                // The maximum occurs at t=1/2, hence
                // max(dist(q(t), [p0, p1] <= 1/2 dist(p1, [p0, p1])).
                //
                // The following takes the square to elide the square root of the Euclidean
                // distance.
                let line = Line::new(p0, p2);
                if line.nearest(p1, 0.).distance_sq <= 4. * TOL_2 {
                    callback.callback(LinePathEl::LineTo(p2));
                } else {
                    let q = QuadBez::new(p0, p1, p2);
                    let params = q.estimate_subdiv(SQRT_TOL);
                    let n = ((0.5 / SQRT_TOL * params.val).ceil() as usize).max(1);
                    let step = 1.0 / (n as f64);
                    for i in 1..n {
                        let u = (i as f64) * step;
                        let t = q.determine_subdiv_t(&params, u);
                        let p = q.eval(t);
                        callback.callback(LinePathEl::LineTo(p));
                    }
                    callback.callback(LinePathEl::LineTo(p2));
                }
                last_pt = p2;
            }
            PathEl::CurveTo(p1, p2, p3) => {
                debug_assert!(!closed, "Expected a `MoveTo` before a `CurveTo`");
                let p0 = last_pt;
                // An upper bound on the shortest distance of any point on the cubic Bezier
                // curve to the line segment [p0, p3] is 3/4 of the maximum of the
                // control-point-to-line-segment distances.
                //
                // With Bernstein weights Bi(t), we have
                // c(t) = B0(t) p0 + B1(t) p1 + B2(t) p2 + B3(t) p3
                // with t from 0 to 1 (inclusive).
                //
                // Through convexivity of the Euclidean distance function and the line segment,
                // we have
                // dist(c(t), [p0, p3]) <= B1(t) dist(p1, [p0, p3]) + B2(t) dist(p2, [p0, p3])
                //                      <= (B1(t) + B2(t)) max(dist(p1, [p0, p3]), dist(p2, [p0, p3]))
                //                       = 3 ((1-t)t^2 + (1-t)^2t) max(dist(p1, [p0, p3]), dist(p2, [p0, p3])).
                //
                // The inner polynomial has its maximum of 1/4 at t=1/2, hence
                // max(dist(c(t), [p0, p3])) <= 3/4 max(dist(p1, [p0, p3]), dist(p2, [p0, p3])).
                //
                // The following takes the square to elide the square root of the Euclidean
                // distance.
                let line = Line::new(p0, p3);
                if f64::max(
                    line.nearest(p1, 0.).distance_sq,
                    line.nearest(p2, 0.).distance_sq,
                ) <= 16. / 9. * TOL_2
                {
                    callback.callback(LinePathEl::LineTo(p3));
                } else {
                    let c = CubicBez::new(p0, p1, p2, p3);
                    let max = flatten_cubic_simd(simd, c, flatten_ctx);

                    for p in &flatten_ctx.flattened_cubics[1..max] {
                        callback.callback(LinePathEl::LineTo(Point::new(p.x as f64, p.y as f64)));
                    }
                }
                last_pt = p3;
            }
            PathEl::ClosePath => {
                closed = true;
                if last_pt != start_pt {
                    callback.callback(LinePathEl::LineTo(start_pt));
                }
            }
        }
    }

    if !closed && last_pt != start_pt {
        callback.callback(LinePathEl::LineTo(start_pt));
    }
}

// The below methods are copied from kurbo and needed to implement flattening of normal quad curves.

/// An approximation to $\int (1 + 4x^2) ^ -0.25 dx$
///
/// This is used for flattening curves.
fn approx_parabola_integral(x: f64) -> f64 {
    const D: f64 = 0.67;
    x / (1.0 - D + (D.powi(4) + 0.25 * x * x).sqrt().sqrt())
}

/// An approximation to the inverse parabola integral.
fn approx_parabola_inv_integral(x: f64) -> f64 {
    const B: f64 = 0.39;
    x * (1.0 - B + (B * B + 0.25 * x * x).sqrt())
}

impl FlattenParamsExt for QuadBez {
    #[inline(always)]
    fn estimate_subdiv(&self, sqrt_tol: f64) -> FlattenParams {
        // Determine transformation to $y = x^2$ parabola.
        let d01 = self.p1 - self.p0;
        let d12 = self.p2 - self.p1;
        let dd = d01 - d12;
        let cross = (self.p2 - self.p0).cross(dd);
        let x0 = d01.dot(dd) * cross.recip();
        let x2 = d12.dot(dd) * cross.recip();
        let scale = (cross / (dd.hypot() * (x2 - x0))).abs();

        // Compute number of subdivisions needed.
        let a0 = approx_parabola_integral(x0);
        let a2 = approx_parabola_integral(x2);
        let val = if scale.is_finite() {
            let da = (a2 - a0).abs();
            let sqrt_scale = scale.sqrt();
            if x0.signum() == x2.signum() {
                da * sqrt_scale
            } else {
                // Handle cusp case (segment contains curvature maximum)
                let xmin = sqrt_tol / sqrt_scale;
                sqrt_tol * da / approx_parabola_integral(xmin)
            }
        } else {
            0.0
        };
        let u0 = approx_parabola_inv_integral(a0);
        let u2 = approx_parabola_inv_integral(a2);
        let uscale = (u2 - u0).recip();
        FlattenParams {
            a0,
            a2,
            u0,
            uscale,
            val,
        }
    }

    #[inline(always)]
    fn determine_subdiv_t(&self, params: &FlattenParams, x: f64) -> f64 {
        let a = params.a0 + (params.a2 - params.a0) * x;
        let u = approx_parabola_inv_integral(a);
        (u - params.u0) * params.uscale
    }
}

trait FlattenParamsExt {
    fn estimate_subdiv(&self, sqrt_tol: f64) -> FlattenParams;
    fn determine_subdiv_t(&self, params: &FlattenParams, x: f64) -> f64;
}

// Everything below is a SIMD implementation of flattening of cubic curves.
// It's a combination of https://gist.github.com/raphlinus/5f4e9feb85fd79bafc72da744571ec0e
// and https://gist.github.com/raphlinus/44e114fef2fd33b889383a60ced0129b.

// TODO(laurenz): Perhaps we should get rid of this in the future and work directly with f32,
// as it's the only reason we have to pull in proc_macros via the `derive` feature of bytemuck.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
struct Point32 {
    x: f32,
    y: f32,
}

struct FlattenParams {
    a0: f64,
    a2: f64,
    u0: f64,
    uscale: f64,
    /// The number of `subdivisions * 2 * sqrt_tol`.
    val: f64,
}

/// This limit was chosen based on the pre-existing GitHub gist.
/// This limit should not be hit in normal operation, but _might_ be hit for very large
/// transforms.
const MAX_QUADS: usize = 16;

/// The context needed for flattening curves.
#[derive(Default, Debug)]
pub struct FlattenCtx {
    // The +4 is to encourage alignment; might be better to be explicit
    even_pts: [Point32; MAX_QUADS + 4],
    odd_pts: [Point32; MAX_QUADS],
    a0: [f32; MAX_QUADS],
    da: [f32; MAX_QUADS],
    u0: [f32; MAX_QUADS],
    uscale: [f32; MAX_QUADS],
    val: [f32; MAX_QUADS],
    n_quads: usize,
    /// Reusable buffer for flattened cubic points.
    flattened_cubics: Vec<Point32>,
}

#[inline(always)]
fn is_finite_simd<S: Simd>(x: f32x4<S>) -> mask32x4<S> {
    let simd = x.simd;

    let x_abs = x.abs();
    let reinterpreted = u32x4::from_bytes(x_abs.to_bytes());
    simd.simd_lt_u32x4(reinterpreted, u32x4::splat(simd, 0x7f80_0000))
}

#[inline(always)]
fn approx_parabola_integral_simd<S: Simd>(x: f32x8<S>) -> f32x8<S> {
    let simd = x.simd;

    const D: f32 = 0.67;
    const D_POWI_4: f32 = 0.201_511_2;

    let temp1 = f32x8::splat(simd, 0.25).madd(x * x, f32x8::splat(simd, D_POWI_4));
    let temp2 = temp1.sqrt();
    let temp3 = temp2.sqrt();
    let temp4 = f32x8::splat(simd, 1.0) - f32x8::splat(simd, D);
    let temp5 = temp4 + temp3;
    x / temp5
}

#[inline(always)]
fn approx_parabola_integral_simd_x4<S: Simd>(x: f32x4<S>) -> f32x4<S> {
    let simd = x.simd;

    const D: f32 = 0.67;
    const D_POWI_4: f32 = 0.201_511_2;

    let temp1 = f32x4::splat(simd, 0.25).madd(x * x, f32x4::splat(simd, D_POWI_4));
    let temp2 = temp1.sqrt();
    let temp3 = temp2.sqrt();
    let temp4 = f32x4::splat(simd, 1.0) - f32x4::splat(simd, D);
    let temp5 = temp4 + temp3;
    x / temp5
}

#[inline(always)]
fn approx_parabola_inv_integral_simd<S: Simd>(x: f32x8<S>) -> f32x8<S> {
    let simd = x.simd;

    const B: f32 = 0.39;
    const ONE_MINUS_B: f32 = 1.0 - B;

    let temp1 = f32x8::splat(simd, B * B);
    let temp2 = f32x8::splat(simd, 0.25).madd(x * x, temp1);
    let temp3 = temp2.sqrt();
    let temp4 = f32x8::splat(simd, ONE_MINUS_B) + temp3;

    x * temp4
}

#[inline(always)]
fn pt_splat_simd<S: Simd>(simd: S, pt: Point32) -> f32x8<S> {
    let p_f64: f64 = bytemuck::cast(pt);
    simd.reinterpret_f32_f64x4(f64x4::splat(simd, p_f64))
}

#[inline(always)]
fn eval_simd<S: Simd>(
    p0: f32x8<S>,
    p1: f32x8<S>,
    p2: f32x8<S>,
    p3: f32x8<S>,
    t: f32x8<S>,
) -> f32x8<S> {
    let mt = 1.0 - t;
    let im0 = p0 * mt * mt * mt;
    let im1 = p1 * mt * mt * 3.0;
    let im2 = p2 * mt * 3.0;
    let im3 = p3.madd(t, im2) * t;

    (im1 + im3).madd(t, im0)
}

#[inline(always)]
fn eval_cubics_simd<S: Simd>(simd: S, c: &CubicBez, n: usize, result: &mut FlattenCtx) {
    result.n_quads = n;
    let dt = 0.5 / n as f32;

    // TODO(laurenz): Perhaps we can SIMDify this better
    let p0p1 = f32x4::from_slice(
        simd,
        &[c.p0.x as f32, c.p0.y as f32, c.p1.x as f32, c.p1.y as f32],
    );
    let p2p3 = f32x4::from_slice(
        simd,
        &[c.p2.x as f32, c.p2.y as f32, c.p3.x as f32, c.p3.y as f32],
    );

    let split_single = |input: f32x4<S>| {
        let t1 = simd.reinterpret_f64_f32x4(input);
        let p0 = simd.zip_low_f64x2(t1, t1);
        let p1 = simd.zip_high_f64x2(t1, t1);

        let p0 = simd.reinterpret_f32_f64x2(p0);
        let p1 = simd.reinterpret_f32_f64x2(p1);

        (f32x8::block_splat(p0), f32x8::block_splat(p1))
    };

    let (p0_128, p1_128) = split_single(p0p1);
    let (p2_128, p3_128) = split_single(p2p3);

    let iota = f32x8::from_slice(simd, &[0.0, 0.0, 2.0, 2.0, 1.0, 1.0, 3.0, 3.0]);
    let step = iota * dt;
    let mut t = step;
    let t_inc = f32x8::splat(simd, 4.0 * dt);

    let even_pts: &mut [f32] = bytemuck::cast_slice_mut(&mut result.even_pts);
    let odd_pts: &mut [f32] = bytemuck::cast_slice_mut(&mut result.odd_pts);

    for i in 0..n.div_ceil(2) {
        let evaluated = eval_simd(p0_128, p1_128, p2_128, p3_128, t);
        let (low, high) = simd.split_f32x8(evaluated);

        even_pts[i * 4..][..4].copy_from_slice(low.as_slice());
        odd_pts[i * 4..][..4].copy_from_slice(high.as_slice());

        t += t_inc;
    }

    even_pts[n * 2..][..8].copy_from_slice(p3_128.as_slice());
}

#[inline(always)]
fn estimate_subdiv_simd<S: Simd>(simd: S, sqrt_tol: f32, ctx: &mut FlattenCtx) {
    let n = ctx.n_quads;

    let even_pts: &mut [f32] = bytemuck::cast_slice_mut(&mut ctx.even_pts);
    let odd_pts: &mut [f32] = bytemuck::cast_slice_mut(&mut ctx.odd_pts);

    for i in 0..n.div_ceil(4) {
        let p0 = f32x8::from_slice(simd, &even_pts[i * 8..][..8]);
        let p_onehalf = f32x8::from_slice(simd, &odd_pts[i * 8..][..8]);
        let p2 = f32x8::from_slice(simd, &even_pts[(i * 8 + 2)..][..8]);
        let x = p0 * -0.5;
        let x1 = p_onehalf.madd(2.0, x);
        let p1 = p2.madd(-0.5, x1);

        odd_pts[(i * 8)..][..8].copy_from_slice(p1.as_slice());

        let d01 = p1 - p0;
        let d12 = p2 - p1;
        let d01x = simd.unzip_low_f32x8(d01, d01);
        let d01y = simd.unzip_high_f32x8(d01, d01);
        let d12x = simd.unzip_low_f32x8(d12, d12);
        let d12y = simd.unzip_high_f32x8(d12, d12);
        let ddx = d01x - d12x;
        let ddy = d01y - d12y;
        let d02x = d01x + d12x;
        let d02y = d01y + d12y;
        // (d02x * ddy) - (d02y * ddx)
        let cross = ddx.madd(-d02y, d02x * ddy);

        let x0_x2_a = {
            let (d01x_low, _) = simd.split_f32x8(d01x);
            let (d12x_low, _) = simd.split_f32x8(d12x);

            simd.combine_f32x4(d12x_low, d01x_low) * ddx
        };
        let temp1 = {
            let (d12y_low, _) = simd.split_f32x8(d12y);
            let (d01y_low, _) = simd.split_f32x8(d01y);

            simd.combine_f32x4(d12y_low, d01y_low)
        };
        let x0_x2_num = temp1.madd(ddy, x0_x2_a);
        let x0_x2 = x0_x2_num / cross;
        let (ddx_low, _) = simd.split_f32x8(ddx);
        let (ddy_low, _) = simd.split_f32x8(ddy);
        let dd_hypot = ddy_low.madd(ddy_low, ddx_low * ddx_low).sqrt();
        let (x0, x2) = simd.split_f32x8(x0_x2);
        let scale_denom = dd_hypot * (x2 - x0);
        let (temp2, _) = simd.split_f32x8(cross);
        let scale = (temp2 / scale_denom).abs();
        let a0_a2 = approx_parabola_integral_simd(x0_x2);
        let (a0, a2) = simd.split_f32x8(a0_a2);
        let da = a2 - a0;
        let da_abs = da.abs();
        let sqrt_scale = scale.sqrt();
        let temp3 = simd.or_i32x4(x0.bitcast(), x2.bitcast());
        let mask = simd.simd_ge_i32x4(temp3, i32x4::splat(simd, 0));
        let noncusp = da_abs * sqrt_scale;
        // TODO: should we skip this if neither is a cusp? Maybe not worth branch prediction cost
        let xmin = sqrt_tol / sqrt_scale;
        let approxint = approx_parabola_integral_simd_x4(xmin);
        let cusp = (sqrt_tol * da_abs) / approxint;
        let val_raw = simd.select_f32x4(mask, noncusp, cusp);
        let finite_mask = is_finite_simd(val_raw);
        let val = simd.select_f32x4(finite_mask, val_raw, f32x4::splat(simd, 0.0));
        let u0_u2 = approx_parabola_inv_integral_simd(a0_a2);
        let (u0, u2) = simd.split_f32x8(u0_u2);
        let uscale_a = u2 - u0;
        let uscale = 1.0 / uscale_a;

        ctx.a0[i * 4..][..4].copy_from_slice(a0.as_slice());
        ctx.da[i * 4..][..4].copy_from_slice(da.as_slice());
        ctx.u0[i * 4..][..4].copy_from_slice(u0.as_slice());
        ctx.uscale[i * 4..][..4].copy_from_slice(uscale.as_slice());
        ctx.val[i * 4..][..4].copy_from_slice(val.as_slice());
    }
}

#[inline(always)]
fn output_lines_simd<S: Simd>(
    simd: S,
    ctx: &mut FlattenCtx,
    i: usize,
    x0: f32,
    dx: f32,
    n: usize,
    start_idx: usize,
) {
    let p0 = pt_splat_simd(simd, ctx.even_pts[i]);
    let p1 = pt_splat_simd(simd, ctx.odd_pts[i]);
    let p2 = pt_splat_simd(simd, ctx.even_pts[i + 1]);

    const IOTA2: [f32; 8] = [0., 0., 1., 1., 2., 2., 3., 3.];
    let iota2 = f32x8::from_slice(simd, IOTA2.as_ref());
    let x = iota2.madd(dx, f32x8::splat(simd, x0));
    let da = f32x8::splat(simd, ctx.da[i]);
    let mut a = da.madd(x, f32x8::splat(simd, ctx.a0[i]));
    let a_inc = 4.0 * dx * da;
    let uscale = f32x8::splat(simd, ctx.uscale[i]);

    let out: &mut [f32] = bytemuck::cast_slice_mut(&mut ctx.flattened_cubics[start_idx..]);

    for j in 0..n.div_ceil(4) {
        let u = approx_parabola_inv_integral_simd(a);
        let t = u.madd(uscale, -ctx.u0[i] * uscale);
        let mt = 1.0 - t;
        let z = p0 * mt * mt;
        let z1 = p1.madd(2.0 * t * mt, z);
        let p = p2.madd(t * t, z1);

        out[j * 8..][..8].copy_from_slice(p.as_slice());

        a += a_inc;
    }
}

#[inline(always)]
fn flatten_cubic_simd<S: Simd>(simd: S, c: CubicBez, ctx: &mut FlattenCtx) -> usize {
    let n_quads = estimate_num_quads(c, TOL as f32);
    eval_cubics_simd(simd, &c, n_quads, ctx);
    let tol = (TOL as f32) * (1.0 - TO_QUAD_TOL);
    let sqrt_tol = tol.sqrt();
    estimate_subdiv_simd(simd, sqrt_tol, ctx);
    let sum: f32 = ctx.val[..n_quads].iter().sum();
    let n = ((0.5 * sum / sqrt_tol).ceil() as usize).max(1);
    let target_len = n + 4;
    if target_len > ctx.flattened_cubics.len() {
        ctx.flattened_cubics.resize(target_len, Point32::default());
    }

    let step = sum / (n as f32);
    let step_recip = 1.0 / step;
    let mut val_sum = 0.0;
    let mut last_n = 0;
    let mut x0base = 0.0;

    for i in 0..n_quads {
        let val = ctx.val[i];
        val_sum += val;
        let this_n = val_sum * step_recip;
        let this_n_next = 1.0 + this_n.floor();
        let dn = this_n_next as usize - last_n;
        if dn > 0 {
            let dx = step / val;
            let x0 = x0base * dx;
            output_lines_simd(simd, ctx, i, x0, dx, dn, last_n);
        }
        x0base = this_n_next - this_n;
        last_n = this_n_next as usize;
    }

    ctx.flattened_cubics[n] = ctx.even_pts[n_quads];

    n + 1
}

#[inline(always)]
fn estimate_num_quads(c: CubicBez, accuracy: f32) -> usize {
    let q_accuracy = (accuracy * TO_QUAD_TOL) as f64;
    let max_hypot2 = 432.0 * q_accuracy * q_accuracy;
    let p1x2 = c.p1.to_vec2() * 3.0 - c.p0.to_vec2();
    let p2x2 = c.p2.to_vec2() * 3.0 - c.p3.to_vec2();
    let err = (p2x2 - p1x2).hypot2();
    let err_div = err / max_hypot2;

    estimate(err_div)
}

const TO_QUAD_TOL: f32 = 0.1;

#[inline(always)]
fn estimate(err_div: f64) -> usize {
    // The original version of this method was:
    // let n_quads = (err_div.powf(1. / 6.0).ceil() as usize).max(1);
    // n_quads.min(MAX_QUADS)
    //
    // Note how we always round up and clamp to the range [1, max_quads]. Since we don't
    // care about the actual fractional value resulting from the powf call we can simply
    // compute this using a precomputed lookup table evaluating 1^6, 2^6, 3^6, etc. and simply
    // comparing if the value is less than or equal to each threshold.

    const LUT: [f64; MAX_QUADS] = [
        1.0, 64.0, 729.0, 4096.0, 15625.0, 46656.0, 117649.0, 262144.0, 531441.0, 1000000.0,
        1771561.0, 2985984.0, 4826809.0, 7529536.0, 11390625.0, 16777216.0,
    ];

    #[expect(clippy::needless_range_loop, reason = "better clarity")]
    for i in 0..MAX_QUADS {
        if err_div <= LUT[i] {
            return i + 1;
        }
    }

    MAX_QUADS
}

#[cfg(test)]
mod tests {
    use crate::flatten_simd::{MAX_QUADS, estimate};

    fn old_estimate(err_div: f64) -> usize {
        let n_quads = (err_div.powf(1. / 6.0).ceil() as usize).max(1);
        n_quads.min(MAX_QUADS)
    }

    // Test is disabled by default since it takes 10-20 seconds to run, even in release mode.
    #[test]
    #[ignore]
    fn accuracy() {
        for i in 0..u32::MAX {
            let num = f32::from_bits(i);

            if num.is_finite() {
                assert_eq!(old_estimate(num as f64), estimate(num as f64), "{num}");
            }
        }
    }
}
