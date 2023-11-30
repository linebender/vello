// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

use crate::cpu_dispatch::CpuBinding;

use super::{
    euler::{CubicParams, EulerParams, EulerSeg, espc_int_approx, espc_int_inv_approx},
    util::{Transform, Vec2, ROBUST_EPSILON},
};
use vello_encoding::{
    math::f16_to_f32, BumpAllocators, ConfigUniform, LineSoup, Monoid, PathBbox, PathMonoid,
    PathTag, Style, DRAW_INFO_FLAGS_FILL_RULE_BIT,
};

// TODO: remove this
macro_rules! log {
    ($($arg:tt)*) => {{
        //println!($($arg)*);
    }};
}

fn to_minus_one_quarter(x: f32) -> f32 {
    // could also be written x.powf(-0.25)
    x.sqrt().sqrt().recip()
}

const D: f32 = 0.67;
fn approx_parabola_integral(x: f32) -> f32 {
    x * to_minus_one_quarter(1.0 - D + (D * D * D * D + 0.25 * x * x))
}

const B: f32 = 0.39;
fn approx_parabola_inv_integral(x: f32) -> f32 {
    x * (1.0 - B + (B * B + 0.5 * x * x)).sqrt()
}

#[derive(Clone, Copy, Default)]
struct SubdivResult {
    val: f32,
    a0: f32,
    a2: f32,
}

fn estimate_subdiv(p0: Vec2, p1: Vec2, p2: Vec2, sqrt_tol: f32) -> SubdivResult {
    let d01 = p1 - p0;
    let d12 = p2 - p1;
    let dd = d01 - d12;
    let cross = (p2.x - p0.x) * dd.y - (p2.y - p0.y) * dd.x;
    let cross_inv = if cross.abs() < 1.0e-9 {
        1.0e9
    } else {
        cross.recip()
    };
    let x0 = d01.dot(dd) * cross_inv;
    let x2 = d12.dot(dd) * cross_inv;
    let scale = (cross / (dd.length() * (x2 - x0))).abs();
    let a0 = approx_parabola_integral(x0);
    let a2 = approx_parabola_integral(x2);
    let mut val = 0.0;
    if scale < 1e9 {
        let da = (a2 - a0).abs();
        let sqrt_scale = scale.sqrt();
        if x0.signum() == x2.signum() {
            val = sqrt_scale;
        } else {
            let xmin = sqrt_tol / sqrt_scale;
            val = sqrt_tol / approx_parabola_integral(xmin);
        }
        val *= da;
    }
    SubdivResult { val, a0, a2 }
}

fn eval_quad(p0: Vec2, p1: Vec2, p2: Vec2, t: f32) -> Vec2 {
    let mt = 1.0 - t;
    p0 * (mt * mt) + (p1 * (mt * 2.0) + p2 * t) * t
}

fn eval_cubic(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: f32) -> Vec2 {
    let mt = 1.0 - t;
    p0 * (mt * mt * mt) + (p1 * (mt * mt * 3.0) + (p2 * (mt * 3.0) + p3 * t) * t) * t
}

fn eval_quad_tangent(p0: Vec2, p1: Vec2, p2: Vec2, t: f32) -> Vec2 {
    let dp0 = 2. * (p1 - p0);
    let dp1 = 2. * (p2 - p1);
    dp0.mix(dp1, t)
}

fn eval_quad_normal(p0: Vec2, p1: Vec2, p2: Vec2, t: f32) -> Vec2 {
    let tangent = eval_quad_tangent(p0, p1, p2, t).normalize();
    Vec2::new(-tangent.y, tangent.x)
}

fn cubic_start_tangent(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2) -> Vec2 {
    let d01 = p1 - p0;
    let d02 = p2 - p0;
    let d03 = p3 - p0;
    if d01.dot(d01) > ROBUST_EPSILON {
        d01
    } else if d02.dot(d02) > ROBUST_EPSILON {
        d02
    } else {
        d03
    }
}

fn cubic_end_tangent(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2) -> Vec2 {
    let d23 = p3 - p2;
    let d13 = p3 - p1;
    let d03 = p3 - p0;
    if d23.dot(d23) > ROBUST_EPSILON {
        d23
    } else if d13.dot(d13) > ROBUST_EPSILON {
        d13
    } else {
        d03
    }
}

fn cubic_start_normal(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2) -> Vec2 {
    let tangent = cubic_start_tangent(p0, p1, p2, p3).normalize();
    Vec2::new(-tangent.y, tangent.x)
}

fn cubic_end_normal(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2) -> Vec2 {
    let tangent = cubic_end_tangent(p0, p1, p2, p3).normalize();
    Vec2::new(-tangent.y, tangent.x)
}

fn cubic_subsegment(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t0: f32, t1: f32) -> CubicPoints {
    let scp0 = eval_cubic(p0, p1, p2, p3, t0);
    let scp3 = eval_cubic(p0, p1, p2, p3, t1);
    // Compute the derivate curve
    let qp0 = 3. * (p1 - p0);
    let qp1 = 3. * (p2 - p1);
    let qp2 = 3. * (p3 - p2);
    let scale = (t1 - t0) * (1.0 / 3.0);
    let scp1 = scp0 + scale * eval_quad(qp0, qp1, qp2, t0);
    let scp2 = scp3 - scale * eval_quad(qp0, qp1, qp2, t1);
    CubicPoints { p0: scp0, p1: scp1, p2: scp2, p3: scp3 }
}

fn check_colinear(p0: Vec2, p1: Vec2, p2: Vec2) -> bool {
    (p1 - p0).cross(p2 - p0).abs() < 1e-9
}

fn write_line(
    line_ix: usize,
    path_ix: u32,
    p0: Vec2,
    p1: Vec2,
    bbox: &mut IntBbox,
    lines: &mut [LineSoup],
) {
    assert!(!p0.is_nan() && !p1.is_nan(), "wrote NaNs: p0: {:?}, p1: {:?}", p0, p1);
    bbox.add_pt(p0);
    bbox.add_pt(p1);
    lines[line_ix] = LineSoup {
        path_ix,
        _padding: Default::default(),
        p0: p0.to_array(),
        p1: p1.to_array(),
    };
}

fn write_line_with_transform(
    line_ix: usize,
    path_ix: u32,
    p0: Vec2,
    p1: Vec2,
    transform: &Transform,
    bbox: &mut IntBbox,
    lines: &mut [LineSoup],
) {
    write_line(
        line_ix,
        path_ix,
        transform.apply(p0),
        transform.apply(p1),
        bbox,
        lines,
    );
}

fn output_line(
    path_ix: u32,
    p0: Vec2,
    p1: Vec2,
    line_ix: &mut usize,
    bbox: &mut IntBbox,
    lines: &mut [LineSoup],
) {
    write_line(*line_ix, path_ix, p0, p1, bbox, lines);
    *line_ix += 1;
}

fn output_line_with_transform(
    path_ix: u32,
    p0: Vec2,
    p1: Vec2,
    transform: &Transform,
    line_ix: &mut usize,
    lines: &mut [LineSoup],
    bbox: &mut IntBbox,
) {
    write_line_with_transform(*line_ix, path_ix, p0, p1, transform, bbox, lines);
    *line_ix += 1;
}

fn output_two_lines_with_transform(
    path_ix: u32,
    p00: Vec2,
    p01: Vec2,
    p10: Vec2,
    p11: Vec2,
    transform: &Transform,
    line_ix: &mut usize,
    lines: &mut [LineSoup],
    bbox: &mut IntBbox,
) {
    write_line_with_transform(*line_ix, path_ix, p00, p01, transform, bbox, lines);
    write_line_with_transform(*line_ix + 1, path_ix, p10, p11, transform, bbox, lines);
    *line_ix += 2;
}

const MAX_QUADS: u32 = 16;

fn flatten_cubic(
    cubic: &CubicPoints,
    path_ix: u32,
    local_to_device: &Transform,
    offset: f32,
    line_ix: &mut usize,
    lines: &mut [LineSoup],
    bbox: &mut IntBbox,
) {
    let (p0, p1, p2, p3, scale, transform) = if offset == 0. {
        (
            local_to_device.apply(cubic.p0),
            local_to_device.apply(cubic.p1),
            local_to_device.apply(cubic.p2),
            local_to_device.apply(cubic.p3),
            1.,
            Transform::identity(),
        )
    } else {
        let t = local_to_device.0;
        let scale = 0.5 * Vec2::new(t[0] + t[3], t[1] - t[2]).length()
            + Vec2::new(t[0] - t[3], t[1] + t[2]).length();
        (
            cubic.p0,
            cubic.p1,
            cubic.p2,
            cubic.p3,
            scale,
            local_to_device.clone(),
        )
    };
    let err_v = (p2 - p1) * 3.0 + p0 - p3;
    let err = err_v.dot(err_v);
    const ACCURACY: f32 = 0.25;
    const Q_ACCURACY: f32 = ACCURACY * 0.1;
    const REM_ACCURACY: f32 = ACCURACY - Q_ACCURACY;
    const MAX_HYPOT2: f32 = 432.0 * Q_ACCURACY * Q_ACCURACY;
    let scaled_sqrt_tol = (REM_ACCURACY / scale).sqrt();
    let mut n_quads = (((err * (1.0 / MAX_HYPOT2)).powf(1.0 / 6.0).ceil() * scale) as u32).max(1);
    n_quads = n_quads.min(MAX_QUADS);

    let mut keep_params = [SubdivResult::default(); MAX_QUADS as usize];
    let mut val = 0.0;
    let mut qp0 = p0;
    let step = (n_quads as f32).recip();
    for i in 0..n_quads {
        let t = (i + 1) as f32 * step;
        let qp2 = eval_cubic(p0, p1, p2, p3, t);
        let mut qp1 = eval_cubic(p0, p1, p2, p3, t - 0.5 * step);
        qp1 = qp1 * 2.0 - (qp0 + qp2) * 0.5;
        let params = estimate_subdiv(qp0, qp1, qp2, scaled_sqrt_tol);
        keep_params[i as usize] = params;
        val += params.val;
        qp0 = qp2;
    }

    let mut n0 = offset * cubic_start_normal(p0, p1, p2, p3);
    let n = ((val * (0.5 / scaled_sqrt_tol)).ceil() as u32).max(1);
    let mut lp0 = p0;
    qp0 = p0;
    let v_step = val / (n as f32);
    let mut n_out = 1;
    let mut val_sum = 0.0;
    for i in 0..n_quads {
        let t = (i + 1) as f32 * step;
        let qp2 = eval_cubic(p0, p1, p2, p3, t);
        let mut qp1 = eval_cubic(p0, p1, p2, p3, t - 0.5 * step);
        qp1 = qp1 * 2.0 - (qp0 + qp2) * 0.5;
        let params = keep_params[i as usize];
        let u0 = approx_parabola_inv_integral(params.a0);
        let u2 = approx_parabola_inv_integral(params.a2);
        let uscale = (u2 - u0).recip();
        let mut val_target = (n_out as f32) * v_step;
        while n_out == n || val_target < val_sum + params.val {
            let (lp1, t1) = if n_out == n {
                (p3, 1.)
            } else {
                let u = (val_target - val_sum) / params.val;
                let a = params.a0 + (params.a2 - params.a0) * u;
                let au = approx_parabola_inv_integral(a);
                let t = (au - u0) * uscale;
                (eval_quad(qp0, qp1, qp2, t), t)
            };
            if offset > 0. {
                let n1 = if lp1 == p3 {
                    cubic_end_normal(p0, p1, p2, p3)
                } else {
                    eval_quad_normal(qp0, qp1, qp2, t1)
                } * offset;
                output_two_lines_with_transform(
                    path_ix,
                    lp0 + n0,
                    lp1 + n1,
                    lp1 - n1,
                    lp0 - n0,
                    &transform,
                    line_ix,
                    lines,
                    bbox,
                );
                n0 = n1;
            } else {
                output_line_with_transform(path_ix, lp0, lp1, &transform, line_ix, lines, bbox);
            }
            n_out += 1;
            val_target += v_step;
            lp0 = lp1;
        }
        val_sum += params.val;
        qp0 = qp2;
    }
}

fn flatten_arc(
    path_ix: u32,
    begin: Vec2,
    end: Vec2,
    center: Vec2,
    angle: f32,
    transform: &Transform,
    line_ix: &mut usize,
    lines: &mut [LineSoup],
    bbox: &mut IntBbox,
) {
    let mut p0 = transform.apply(begin);
    let mut r = begin - center;
    let tol: f32 = 0.5;
    let radius = tol.max((p0 - transform.apply(center)).length());
    let x = 1. - tol / radius;
    let theta = (2. * x * x - 1.).clamp(-1., 1.).acos();
    const MAX_LINES: u32 = 1000;
    let n_lines = if theta <= ROBUST_EPSILON {
        MAX_LINES
    } else {
        MAX_LINES.min((std::f32::consts::TAU / theta).ceil() as u32)
    };

    let th = angle / (n_lines as f32);
    let c = th.cos();
    let s = th.sin();
    let rot = Transform([c, -s, s, c, 0., 0.]);

    for _ in 0..(n_lines - 1) {
        r = rot.apply(r);
        let p1 = transform.apply(center + r);
        output_line(path_ix, p0, p1, line_ix, bbox, lines);
        p0 = p1;
    }
    let p1 = transform.apply(end);
    output_line(path_ix, p0, p1, line_ix, bbox, lines);
}

fn flatten_euler(
    cubic: &CubicPoints,
    path_ix: u32,
    local_to_device: &Transform,
    offset: f32,
    is_line: bool,
    line_ix: &mut usize,
    lines: &mut [LineSoup],
    bbox: &mut IntBbox,
)  {
    log!("@@@ flatten_euler: {:#?}", cubic);
	// Flatten in local coordinates if this is a stroke. Flatten in device space otherwise.
    let (p0, p1, p2, p3, scale, transform) = if offset == 0. {
        (
            local_to_device.apply(cubic.p0),
            local_to_device.apply(cubic.p1),
            local_to_device.apply(cubic.p2),
            local_to_device.apply(cubic.p3),
            1.,
            Transform::identity(),
        )
    } else {
        let t = local_to_device.0;
        let scale = 0.5 * Vec2::new(t[0] + t[3], t[1] - t[2]).length()
            + Vec2::new(t[0] - t[3], t[1] + t[2]).length();
        (
            cubic.p0,
            cubic.p1,
            cubic.p2,
            cubic.p3,
            scale,
            local_to_device.clone(),
        )
    };

    // Special-case lines.
    // We still have to handle colinear cubic parameters. We are special casing the line-to
    // encoding because floating point errors in the degree raise can cause some line-tos to slip
    // through the epsilon threshold in check_colinear.
    //if check_colinear(p0, p1, p3) && check_colinear(p0, p2, p3) {
    if is_line {
        let tan = p3 - p0;
        if tan.length() > ROBUST_EPSILON {
            let tan_norm = tan.normalize();
            let n = Vec2::new(-tan_norm.y, tan_norm.x);
            let lp0 = p0 + n * offset;
            let lp1 = p3 + n * offset;
            let l0 = if offset > 0. { lp0 } else { lp1 };
            let l1 = if offset > 0. { lp1 } else { lp0 };
            log!("@@@ output line: {:?}, {:?}, tan: {:?}", l0, l1, tan);
            output_line_with_transform(path_ix, l0, l1, &transform, line_ix, lines, bbox);
        } else {
            log!("@@@ drop line: {:?}, {:?}, tan: {:?}", p0, p3, tan);
        }
        return;
    }

    // Special-case colinear cubic segments
    // TODO: clean this up
    if check_colinear(p0, p1, p3) && check_colinear(p0, p2, p3) && check_colinear(p0, p1, p2) && check_colinear(p1, p2, p3) {
        let (start, end) = {
            let distances = [
                ((p1 - p0).length(), p1, p0),
                ((p2 - p0).length(), p2, p0),
                ((p3 - p0).length(), p3, p0),
                ((p2 - p1).length(), p2, p1),
                ((p3 - p1).length(), p3, p1),
                ((p3 - p2).length(), p3, p2),
            ];
            let mut longest = distances[0];
            for d in &distances[1..] {
                if d.0 > longest.0 {
                    longest = *d;
                }
            }
            (longest.1, longest.2)
        };
        let tan = end - start;
        if tan.length() > ROBUST_EPSILON {
            let tan_norm = tan.normalize();
            let n = Vec2::new(-tan_norm.y, tan_norm.x);
            let lp0 = start + n * offset;
            let lp1 = end + n * offset;
            let l0 = if offset > 0. { lp0 } else { lp1 };
            let l1 = if offset > 0. { lp1 } else { lp0 };
            log!("@@@ output line: {:?}, {:?}, tan: {:?}", l0, l1, tan);
            output_line_with_transform(path_ix, l0, l1, &transform, line_ix, lines, bbox);
        } else {
            log!("@@@ drop line: {:?}, {:?}, tan: {:?}", start, end, tan);
        }
        return;
    }

	let tol: f32 = 0.01;
	let scaled_sqrt_tol = (tol / scale).sqrt();
    let mut t0_u: u32 = 0;
	let mut dt: f32 = 1.;

	loop {
        if dt < ROBUST_EPSILON {
            break;
        }
		let t0 = (t0_u as f32) * dt;
        if t0 == 1. {
            break;
        }
        log!("@@@ loop1: t0: {t0}, dt: {dt}");
        loop {
            let t1 = t0 + dt;
            // Subdivide into cubics
            let subcubic = cubic_subsegment(p0, p1, p2, p3, t0, t1);
            let cubic_params = CubicParams::from_cubic(subcubic.p0, subcubic.p1, subcubic.p2, subcubic.p3);
            let est_err = cubic_params.est_euler_err();
            let err = est_err * (subcubic.p0 - subcubic.p3).length();
            log!("@@@   loop2: sub:{:?}, {:?} t0: {t0}, t1: {t1}, dt: {dt}, est_err: {est_err}, err: {err}", subcubic, cubic_params);
            if err <= scaled_sqrt_tol {
                log!("@@@   error within tolerance");
                t0_u += 1;
                let shift = t0_u.trailing_zeros();
                t0_u >>= shift;
                dt *= (1 << shift) as f32;
                let euler_params = EulerParams::from_angles(cubic_params.th0, cubic_params.th1);
                let es = EulerSeg::from_params(subcubic.p0, subcubic.p3, euler_params);

                let es_scale = (es.p0 - es.p1).length();
                let (k0, k1) = (es.params.k0 - 0.5 * es.params.k1, es.params.k1);

               	// compute forward integral to determine number of subdivisions
				let dist_scaled = offset * scale / es_scale;
				let a = -2.0 * dist_scaled * k1;
				let b = -1.0 - 2.0 * dist_scaled * k0;
				let int0 = espc_int_approx(b);
				let int1 = espc_int_approx(a + b);
				let integral = int1 - int0;
				let k_peak = k0 - k1 * b / a;
				let integrand_peak = (k_peak * (k_peak * dist_scaled + 1.0)).abs().sqrt();
				let scaled_int = integral * integrand_peak / a;
				let n_frac = 0.5 * (es_scale / scaled_sqrt_tol).sqrt() * scaled_int;
				let n = n_frac.ceil();

                // Flatten line segments
                log!("@@@   loop2: lines: {n}");
                // TODO: make all computation above robust and uncomment this assertion
                //assert!(!n.is_nan());
                if n.is_nan() {
                    // Skip the segment if `n` is NaN. This is for debugging purposes only
                    log!("@@@   NaN: parameters:\n  es: {:#?}\n  k0: {k0}, k1: {k1}\n  dist_scaled: {dist_scaled}\n  es_scale: {es_scale}\n  a: {a}\n  b: {b}\n  int0: {int0}, int1: {int1}, integral: {integral}\n  k_peak: {k_peak}\n  integrand_peak: {integrand_peak}\n  scaled_int: {scaled_int}\n  n_frac:  {n_frac}", es);
                } else {
                    let mut lp0 = es.eval_with_offset(0., offset);
                    for i in 0..n as usize {
                        let t = (i + 1) as f32 / n;
                        let inv = espc_int_inv_approx(integral * t + int0);
                        let s = (inv - b) / a;
                        let lp1 = es.eval_with_offset(s, offset);
                        let l0 = if offset > 0. { lp0 } else { lp1 };
                        let l1 = if offset > 0. { lp1 } else { lp0 };
                        output_line_with_transform(path_ix, l0, l1, &transform, line_ix, lines, bbox);
                        lp0 = lp1;
                    }
                }
                break;
            }
            t0_u = t0_u.saturating_mul(2);
            dt *= 0.5;
        }
	}
}

fn draw_cap(
    path_ix: u32,
    cap_style: u32,
    point: Vec2,
    cap0: Vec2,
    cap1: Vec2,
    offset_tangent: Vec2,
    transform: &Transform,
    line_ix: &mut usize,
    lines: &mut [LineSoup],
    bbox: &mut IntBbox,
) {
    if cap_style == Style::FLAGS_CAP_BITS_ROUND {
        flatten_arc(
            path_ix,
            cap0,
            cap1,
            point,
            std::f32::consts::PI,
            transform,
            line_ix,
            lines,
            bbox,
        );
        return;
    }

    let mut start = cap0;
    let mut end = cap1;
    if cap_style == Style::FLAGS_CAP_BITS_SQUARE {
        let v = offset_tangent;
        let p0 = start + v;
        let p1 = end + v;
        output_line_with_transform(path_ix, start, p0, transform, line_ix, lines, bbox);
        output_line_with_transform(path_ix, p1, end, transform, line_ix, lines, bbox);
        start = p0;
        end = p1;
    }
    output_line_with_transform(path_ix, start, end, transform, line_ix, lines, bbox);
}

fn draw_join(
    path_ix: u32,
    style_flags: u32,
    p0: Vec2,
    tan_prev: Vec2,
    tan_next: Vec2,
    n_prev: Vec2,
    n_next: Vec2,
    transform: &Transform,
    line_ix: &mut usize,
    lines: &mut [LineSoup],
    bbox: &mut IntBbox,
) {
    let mut front0 = p0 + n_prev;
    let front1 = p0 + n_next;
    let mut back0 = p0 - n_next;
    let back1 = p0 - n_prev;

    let cr = tan_prev.x * tan_next.y - tan_prev.y * tan_next.x;
    let d = tan_prev.dot(tan_next);

    match style_flags & Style::FLAGS_JOIN_MASK {
        Style::FLAGS_JOIN_BITS_BEVEL => {
            output_two_lines_with_transform(
                path_ix, front0, front1, back0, back1, transform, line_ix, lines, bbox,
            );
        }
        Style::FLAGS_JOIN_BITS_MITER => {
            let hypot = cr.hypot(d);
            let miter_limit = f16_to_f32((style_flags & Style::MITER_LIMIT_MASK) as u16);

            if 2. * hypot < (hypot + d) * miter_limit * miter_limit && cr != 0. {
                let is_backside = cr > 0.;
                let fp_last = if is_backside { back1 } else { front0 };
                let fp_this = if is_backside { back0 } else { front1 };
                let p = if is_backside { back0 } else { front0 };

                let v = fp_this - fp_last;
                let h = (tan_prev.x * v.y - tan_prev.y * v.x) / cr;
                let miter_pt = fp_this - tan_next * h;

                output_line_with_transform(path_ix, p, miter_pt, transform, line_ix, lines, bbox);

                if is_backside {
                    back0 = miter_pt;
                } else {
                    front0 = miter_pt;
                }
            }
            output_two_lines_with_transform(
                path_ix, front0, front1, back0, back1, transform, line_ix, lines, bbox,
            );
        }
        Style::FLAGS_JOIN_BITS_ROUND => {
            let (arc0, arc1, other0, other1) = if cr > 0. {
                (back0, back1, front0, front1)
            } else {
                (front0, front1, back0, back1)
            };
            flatten_arc(
                path_ix,
                arc0,
                arc1,
                p0,
                cr.atan2(d).abs(),
                transform,
                line_ix,
                lines,
                bbox,
            );
            output_line_with_transform(path_ix, other0, other1, transform, line_ix, lines, bbox);
        }
        _ => unreachable!(),
    }
}

fn read_f32_point(ix: u32, pathdata: &[u32]) -> Vec2 {
    let x = f32::from_bits(pathdata[ix as usize]);
    let y = f32::from_bits(pathdata[ix as usize + 1]);
    Vec2 { x, y }
}

fn read_i16_point(ix: u32, pathdata: &[u32]) -> Vec2 {
    let raw = pathdata[ix as usize];
    let x = (((raw << 16) as i32) >> 16) as f32;
    let y = ((raw as i32) >> 16) as f32;
    Vec2 { x, y }
}

#[derive(Debug)]
struct IntBbox {
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
}

impl Default for IntBbox {
    fn default() -> Self {
        IntBbox {
            x0: 0x7fff_ffff,
            y0: 0x7fff_ffff,
            x1: -0x8000_0000,
            y1: -0x8000_0000,
        }
    }
}

impl IntBbox {
    fn add_pt(&mut self, pt: Vec2) {
        self.x0 = self.x0.min(pt.x.floor() as i32);
        self.y0 = self.y0.min(pt.y.floor() as i32);
        self.x1 = self.x1.max(pt.x.ceil() as i32);
        self.y1 = self.y1.max(pt.y.ceil() as i32);
    }
}

struct PathTagData {
    tag_byte: u8,
    monoid: PathMonoid,
}

fn compute_tag_monoid(ix: usize, pathtags: &[u32], tag_monoids: &[PathMonoid]) -> PathTagData {
    let tag_word = pathtags[ix >> 2];
    let shift = (ix & 3) * 8;
    let mut tm = PathMonoid::new(tag_word & ((1 << shift) - 1));
    let tag_byte = ((tag_word >> shift) & 0xff) as u8;
    if tag_byte != 0 {
        tm = tag_monoids[ix >> 2].combine(&tm);
    }
    PathTagData {
        tag_byte,
        monoid: tm,
    }
}

#[derive(Debug)]
struct CubicPoints {
    p0: Vec2,
    p1: Vec2,
    p2: Vec2,
    p3: Vec2,
}

fn read_path_segment(tag: &PathTagData, is_stroke: bool, pathdata: &[u32]) -> CubicPoints {
    let mut p0;
    let mut p1;
    let mut p2 = Vec2::default();
    let mut p3 = Vec2::default();

    let mut seg_type = tag.tag_byte & PATH_TAG_SEG_TYPE;
    let pathseg_offset = tag.monoid.pathseg_offset;
    let is_stroke_cap_marker = is_stroke && (tag.tag_byte & PathTag::SUBPATH_END_BIT) != 0;
    let is_open = seg_type == PATH_TAG_QUADTO;

    if (tag.tag_byte & PATH_TAG_F32) != 0 {
        p0 = read_f32_point(pathseg_offset, pathdata);
        p1 = read_f32_point(pathseg_offset + 2, pathdata);
        if seg_type >= PATH_TAG_QUADTO {
            p2 = read_f32_point(pathseg_offset + 4, pathdata);
            if seg_type == PATH_TAG_CUBICTO {
                p3 = read_f32_point(pathseg_offset + 6, pathdata);
            }
        }
    } else {
        p0 = read_i16_point(pathseg_offset, pathdata);
        p1 = read_i16_point(pathseg_offset + 1, pathdata);
        if seg_type >= PATH_TAG_QUADTO {
            p2 = read_i16_point(pathseg_offset + 2, pathdata);
            if seg_type == PATH_TAG_CUBICTO {
                p3 = read_i16_point(pathseg_offset + 3, pathdata);
            }
        }
    }

    if is_stroke_cap_marker && is_open {
        p0 = p1;
        p1 = p2;
        seg_type = PATH_TAG_LINETO;
    }

    // Degree-raise
    if seg_type == PATH_TAG_LINETO {
        p3 = p1;
        p2 = p3.mix(p0, 1.0 / 3.0);
        p1 = p0.mix(p3, 1.0 / 3.0);
    } else if seg_type == PATH_TAG_QUADTO {
        p3 = p2;
        p2 = p1.mix(p2, 1.0 / 3.0);
        p1 = p1.mix(p0, 1.0 / 3.0);
    }

    CubicPoints { p0, p1, p2, p3 }
}

struct NeighboringSegment {
    do_join: bool,
    tangent: Vec2,
}

fn read_neighboring_segment(
    ix: usize,
    pathtags: &[u32],
    pathdata: &[u32],
    tag_monoids: &[PathMonoid],
) -> NeighboringSegment {
    let tag = compute_tag_monoid(ix, pathtags, tag_monoids);
    let pts = read_path_segment(&tag, true, pathdata);

    let is_closed = (tag.tag_byte & PATH_TAG_SEG_TYPE) == PATH_TAG_LINETO;
    let is_stroke_cap_marker = (tag.tag_byte & PathTag::SUBPATH_END_BIT) != 0;
    let do_join = !is_stroke_cap_marker || is_closed;
    let tangent = cubic_start_tangent(pts.p0, pts.p1, pts.p2, pts.p3);
    NeighboringSegment { do_join, tangent }
}

const WG_SIZE: usize = 256;

const PATH_TAG_SEG_TYPE: u8 = 3;
const PATH_TAG_PATH: u8 = 0x10;
const PATH_TAG_LINETO: u8 = 1;
const PATH_TAG_QUADTO: u8 = 2;
const PATH_TAG_CUBICTO: u8 = 3;
const PATH_TAG_F32: u8 = 8;

fn flatten_main(
    n_wg: u32,
    config: &ConfigUniform,
    scene: &[u32],
    tag_monoids: &[PathMonoid],
    path_bboxes: &mut [PathBbox],
    bump: &mut BumpAllocators,
    lines: &mut [LineSoup],
) {
    let mut line_ix = 0;
    let pathtags = &scene[config.layout.path_tag_base as usize..];
    let pathdata = &scene[config.layout.path_data_base as usize..];

    for ix in 0..n_wg as usize * WG_SIZE {
        let mut bbox = IntBbox::default();
        let tag = compute_tag_monoid(ix, pathtags, tag_monoids);
        let path_ix = tag.monoid.path_ix;
        let style_ix = tag.monoid.style_ix;
        let trans_ix = tag.monoid.trans_ix;
        let style_flags = scene[(config.layout.style_base + style_ix) as usize];
        if (tag.tag_byte & PATH_TAG_PATH) != 0 {
            let out = &mut path_bboxes[path_ix as usize];
            out.draw_flags = if (style_flags & Style::FLAGS_FILL_BIT) == 0 {
                0
            } else {
                DRAW_INFO_FLAGS_FILL_RULE_BIT
            };
            out.trans_ix = trans_ix;
        }

        let seg_type = tag.tag_byte & PATH_TAG_SEG_TYPE;
        if seg_type != 0 {
            let is_stroke = (style_flags & Style::FLAGS_STYLE_BIT) != 0;
            let transform = Transform::read(config.layout.transform_base, trans_ix, scene);
            let pts = read_path_segment(&tag, is_stroke, pathdata);

            if is_stroke {
                let linewidth =
                    f32::from_bits(scene[(config.layout.style_base + style_ix + 1) as usize]);
                let offset = 0.5 * linewidth;

                let is_open = seg_type != PATH_TAG_LINETO;
                let is_stroke_cap_marker = (tag.tag_byte & PathTag::SUBPATH_END_BIT) != 0;
                if is_stroke_cap_marker {
                    if is_open {
                        // Draw start cap
                        let tangent = cubic_start_tangent(pts.p0, pts.p1, pts.p2, pts.p3);
                        let offset_tangent = offset * tangent.normalize();
                        let n = Vec2::new(-offset_tangent.y, offset_tangent.x);
                        draw_cap(
                            path_ix,
                            (style_flags & Style::FLAGS_START_CAP_MASK) >> 2,
                            pts.p0,
                            pts.p0 - n,
                            pts.p0 + n,
                            -offset_tangent,
                            &transform,
                            &mut line_ix,
                            lines,
                            &mut bbox,
                        );
                    } else {
                        // Don't draw anything if the path is closed.
                    }
                } else {
                    let is_line = seg_type == PATH_TAG_LINETO;
                    // Render offset curves
                    flatten_euler(
                        &pts,
                        path_ix,
                        &transform,
                        offset,
                        is_line,
                        &mut line_ix,
                        lines,
                        &mut bbox,
                    );
                    flatten_euler(
                        &pts,
                        path_ix,
                        &transform,
                        -offset,
                        is_line,
                        &mut line_ix,
                        lines,
                        &mut bbox,
                    );

                    // Read the neighboring segment.
                    let neighbor =
                        read_neighboring_segment(ix + 1, pathtags, pathdata, tag_monoids);
                    let tan_prev = cubic_end_tangent(pts.p0, pts.p1, pts.p2, pts.p3);
                    let tan_next = neighbor.tangent;

                    // TODO: add NaN assertions to CPU shaders PR (when writing lines)
                    // TODO: not all zero-length segments are getting filtered out
                    // TODO: this is a hack. How to handle caps on degenerate stroke?
                    // TODO: debug tricky stroke by isolation
                    let tan_prev = if tan_prev.length_squared() < ROBUST_EPSILON {
                        Vec2::new(0.01, 0.)
                    } else {
                        tan_prev
                    };
                    let tan_next = if tan_next.length_squared() < ROBUST_EPSILON {
                        Vec2::new(0.01, 0.)
                    } else {
                        tan_next
                    };

                    let offset_tangent = offset * tan_prev.normalize();
                    let n_prev = Vec2::new(-offset_tangent.y, offset_tangent.x);
                    let tan_next_norm = tan_next.normalize();
                    let n_next = offset * Vec2::new(-tan_next_norm.y, tan_next_norm.x);
                    log!("@ tan_prev: {:#?}", tan_prev);
                    log!("@ tan_next: {:#?}", tan_next);
                    if neighbor.do_join {
                        draw_join(
                            path_ix,
                            style_flags,
                            pts.p3,
                            tan_prev,
                            tan_next,
                            n_prev,
                            n_next,
                            &transform,
                            &mut line_ix,
                            lines,
                            &mut bbox,
                        );
                    } else {
                        // Draw end cap.
                        draw_cap(
                            path_ix,
                            style_flags & Style::FLAGS_END_CAP_MASK,
                            pts.p3,
                            pts.p3 + n_prev,
                            pts.p3 - n_prev,
                            offset_tangent,
                            &transform,
                            &mut line_ix,
                            lines,
                            &mut bbox,
                        );
                    }
                }
            } else {
                flatten_cubic(
                    &pts,
                    path_ix,
                    &transform,
                    /*offset*/ 0.,
                    &mut line_ix,
                    lines,
                    &mut bbox,
                );
            }
        }

        if (path_ix as usize) < path_bboxes.len() && (bbox.x1 > bbox.x0 || bbox.y1 > bbox.y0) {
            let out = &mut path_bboxes[path_ix as usize];
            out.x0 = out.x0.min(bbox.x0);
            out.y0 = out.y0.min(bbox.y0);
            out.x1 = out.x1.max(bbox.x1);
            out.y1 = out.y1.max(bbox.y1);
        }
    }
    bump.lines = line_ix as u32;
}

pub fn flatten(n_wg: u32, resources: &[CpuBinding]) {
    let config = resources[0].as_typed();
    let scene = resources[1].as_slice();
    let tag_monoids = resources[2].as_slice();
    let mut path_bboxes = resources[3].as_slice_mut();
    let mut bump = resources[4].as_typed_mut();
    let mut lines = resources[5].as_slice_mut();
    flatten_main(
        n_wg,
        &config,
        &scene,
        &tag_monoids,
        &mut path_bboxes,
        &mut bump,
        &mut lines,
    );
}
