// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Flatten curves to lines

#import config
#import drawtag
#import pathtag
#import segment
#import cubic
#import bump

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> scene: array<u32>;

@group(0) @binding(2)
var<storage> tag_monoids: array<TagMonoid>;

struct AtomicPathBbox {
    x0: atomic<i32>,
    y0: atomic<i32>,
    x1: atomic<i32>,
    y1: atomic<i32>,
    draw_flags: u32,
    trans_ix: u32,
}

@group(0) @binding(3)
var<storage, read_write> path_bboxes: array<AtomicPathBbox>;

@group(0) @binding(4)
var<storage, read_write> bump: BumpAllocators;

@group(0) @binding(5)
var<storage, read_write> lines: array<LineSoup>;

struct SubdivResult {
    val: f32,
    a0: f32,
    a2: f32,
}

const D = 0.67;
fn approx_parabola_integral(x: f32) -> f32 {
    return x * inverseSqrt(sqrt(1.0 - D + (D * D * D * D + 0.25 * x * x)));
}

const B = 0.39;
fn approx_parabola_inv_integral(x: f32) -> f32 {
    return x * sqrt(1.0 - B + (B * B + 0.5 * x * x));
}

// Functions for Euler spirals

struct CubicParams {
    th0: f32,
    th1: f32,
    chord_len: f32,
    err: f32,
}

struct EulerParams {
    th0: f32,
    // th1 need not be explicitly stored, as it can be derived from k0 - th0
    k0: f32,
    k1: f32,
    ch: f32,
}

struct EulerSeg {
    p0: vec2f,
    p1: vec2f,
    params: EulerParams,
}

// Threshold below which a derivative is considered too small.
const DERIV_THRESH: f32 = 1e-6;
const DERIV_THRESH_SQUARED: f32 = DERIV_THRESH * DERIV_THRESH;
// Amount to nudge t when derivative is near-zero.
const DERIV_EPS: f32 = 1e-6;
// Limit for subdivision of cubic Béziers.
const SUBDIV_LIMIT: f32 = 1.0 / 65536.0;
// Robust ESPC computation: below this value, treat curve as circular arc
const K1_THRESH: f32 = 1e-3;
// Robust ESPC: below this value, evaluate ES rather than parallel curve
const DIST_THRESH: f32 = 1e-3;
// Threshold for tangents to be considered near zero length
const TANGENT_THRESH: f32 = 1e-6;

/// Compute cubic parameters from endpoints and derivatives.
fn cubic_from_points_derivs(p0: vec2f, p1: vec2f, q0: vec2f, q1: vec2f, dt: f32) -> CubicParams {
    let chord = p1 - p0;
    let chord_squared = dot(chord, chord);
    let chord_len = sqrt(chord_squared);
    if chord_squared < DERIV_THRESH_SQUARED {
        let chord_err = sqrt((9. / 32.0) * (dot(q0, q0) + dot(q1, q1))) * dt;
        return CubicParams(0.0, 0.0, DERIV_THRESH, chord_err);
    }
    let scale = dt / chord_squared;
    let h0 = vec2(q0.x * chord.x + q0.y * chord.y, q0.y * chord.x - q0.x * chord.y);
    let th0 = atan2(h0.y, h0.x);
    let d0 = length(h0) * scale;
    let h1 = vec2(q1.x * chord.x + q1.y * chord.y, q1.x * chord.y - q1.y * chord.x);
    let th1 = atan2(h1.y, h1.x);
    let d1 = length(h1) * scale;

    // Estimate error of geometric Hermite interpolation to Euler spiral.
    let cth0 = cos(th0);
    let cth1 = cos(th1);
    var err = 2.0;
    if cth0 * cth1 >= 0.0 {
        let e0 = (2. / 3.) / max(1.0 + cth0, 1e-9);
        let e1 = (2. / 3.) / max(1.0 + cth1, 1e-9);
        let s0 = sin(th0);
        let s1 = sin(th1);
        let s01 = cth0 * s1 + cth1 * s0;
        let amin = 0.15 * (2. * e0 * s0 + 2. * e1 * s1 - e0 * e1 * s01);
        let a = 0.15 * (2. * d0 * s0 + 2. * d1 * s1 - d0 * d1 * s01);
        let aerr = abs(a - amin);
        let symm = abs(th0 + th1);
        let asymm = abs(th0 - th1);
        let dist = length(vec2(d0 - e0, d1 - e1));
        let symm2 = symm * symm;
        let ctr = (4.625e-6 * symm * symm2 + 7.5e-3 * asymm) * symm2;
        let halo = (5e-3 * symm + 7e-2 * asymm) * dist;
        err = ctr + 1.55 * aerr + halo;
    }
    err *= chord_len;
    return CubicParams(th0, th1, chord_len, err);
}

fn es_params_from_angles(th0: f32, th1: f32) -> EulerParams {
    let k0 = th0 + th1;
    let dth = th1 - th0;
    let d2 = dth * dth;
    let k2 = k0 * k0;
    var a = 6.0;
    a -= d2 * (1. / 70.);
    a -= (d2 * d2) * (1. / 10780.);
    a += (d2 * d2 * d2) * 2.769178184818219e-07;
    let b = -0.1 + d2 * (1. / 4200.) + d2 * d2 * 1.6959677820260655e-05;
    let c = -1. / 1400. + d2 * 6.84915970574303e-05 - k2 * 7.936475029053326e-06;
    a += (b + c * k2) * k2;
    let k1 = dth * a;

    // calculation of chord
    var ch = 1.0;
    ch -= d2 * (1. / 40.);
    ch += (d2 * d2) * 0.00034226190482569864;
    ch -= (d2 * d2 * d2) * 1.9349474568904524e-06;
    let b_ = -1. / 24. + d2 * 0.0024702380951963226 - d2 * d2 * 3.7297408997537985e-05;
    let c_ = 1. / 1920. - d2 * 4.87350869747975e-05 - k2 * 3.1001936068463107e-06;
    ch += (b_ + c_ * k2) * k2;
    return EulerParams(th0, k0, k1, ch);
}

fn es_params_eval_th(params: EulerParams, t: f32) -> f32 {
    return (params.k0 + 0.5 * params.k1 * (t - 1.0)) * t - params.th0;
}

// Integrate Euler spiral.
fn integ_euler_10(k0: f32, k1: f32) -> vec2f {
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
    var u = 1.;
    u -= (1. / 24.) * t2_2 + (1. / 160.) * t2_4;
    u += (1. / 1920.) * t4_4 + (1. / 10752.) * t4_6 + (1. / 55296.) * t4_8;
    u -= (1. / 322560.) * t6_6 + (1. / 1658880.) * t6_8;
    u += (1. / 92897280.) * t8_8;
    var v = (1. / 12.) * t1_2;
    v -= (1. / 480.) * t3_4 + (1. / 2688.) * t3_6;
    v += (1. / 53760.) * t5_6 + (1. / 276480.) * t5_8;
    v -= (1. / 11612160.) * t7_8;
    return vec2(u, v);
}

fn es_params_eval(params: EulerParams, t: f32) -> vec2f {
    let thm = es_params_eval_th(params, t * 0.5);
    let k0 = params.k0;
    let k1 = params.k1;
    let uv = integ_euler_10((k0 + k1 * (0.5 * t - 0.5)) * t, k1 * t * t);
    let scale = t / params.ch;
    let s = scale * sin(thm);
    let c = scale * cos(thm);
    let x = uv.x * c - uv.y * s;
    let y = -uv.y * c - uv.x * s;
    return vec2(x, y);
}

fn es_params_eval_with_offset(params: EulerParams, t: f32, offset: f32) -> vec2f {
    let th = es_params_eval_th(params, t);
    let v = offset * vec2f(sin(th), cos(th));
    return es_params_eval(params, t) + v;
}

fn es_seg_from_params(p0: vec2f, p1: vec2f, params: EulerParams) -> EulerSeg {
    return EulerSeg(p0, p1, params);
}

// Note: offset provided is scaled so that 1 = chord length
fn es_seg_eval_with_offset(es: EulerSeg, t: f32, normalized_offset: f32) -> vec2f {
    let chord = es.p1 - es.p0;
    let xy = es_params_eval_with_offset(es.params, t, normalized_offset);
    return es.p0 + vec2f(chord.x * xy.x - chord.y * xy.y, chord.x * xy.y + chord.y * xy.x);
}

fn pow_1_5_signed(x: f32) -> f32 {
    return x * sqrt(abs(x));
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
const QUAD_W1: f32 = 0.5 * QUAD_B1 / QUAD_A1;
const QUAD_V1: f32 = 1.0 / QUAD_A1;
const QUAD_U1: f32 = QUAD_W1 * QUAD_W1 - QUAD_C1 / QUAD_A1;
const QUAD_W2: f32 = 0.5 * QUAD_B2 / QUAD_A2;
const QUAD_V2: f32 = 1.0 / QUAD_A2;
const QUAD_U2: f32 = QUAD_W2 * QUAD_W2 - QUAD_C2 / QUAD_A2;
const FRAC_PI_4: f32 = 0.7853981633974483;
const CBRT_9_8: f32 = 1.040041911525952;

fn espc_int_approx(x: f32) -> f32 {
    let y = abs(x);
    var a: f32;
    if y < BREAK1 {
        a = sin(SIN_SCALE * y) * (1.0 / SIN_SCALE);
    } else if y < BREAK2 {
        a = (sqrt(8.0) / 3.0) * pow_1_5_signed(y - 1.0) + FRAC_PI_4;
    } else {
        let abc = select(vec3(QUAD_A2, QUAD_B2, QUAD_C2), vec3(QUAD_A1, QUAD_B1, QUAD_C1), y < BREAK3);
        a = (abc.x * y + abc.y) * y + abc.z;
    };
    return a * sign(x);
}

fn espc_int_inv_approx(x: f32) -> f32 {
    let y = abs(x);
    var a: f32;
    if y < 0.7010707591262915 {
        a = asin(y * SIN_SCALE) * (1.0 / SIN_SCALE);
    } else if y < 0.903249293595206 {
        let b = y - FRAC_PI_4;
        let u = pow(abs(b), 2. / 3.) * sign(b);
        a = u * CBRT_9_8 + 1.0;
    } else {
        let uvw = select(vec3(QUAD_U2, QUAD_V2, QUAD_W2), vec3(QUAD_U1, QUAD_V1, QUAD_W1), y < 2.038857793595206);
        a = sqrt(uvw.x + uvw.y * y) - uvw.z;
    }
    return a * sign(x);
}

struct PointDeriv {
    point: vec2f,
    deriv: vec2f,
}

fn eval_cubic_and_deriv(p0: vec2f, p1: vec2f, p2: vec2f, p3: vec2f, t: f32) -> PointDeriv {
    let m = 1.0 - t;
    let mm = m * m;
    let mt = m * t;
    let tt = t * t;
    let p = p0 * (mm * m) + (p1 * (3.0 * mm) + p2 * (3.0 * mt) + p3 * tt) * t;
    let q = (p1 - p0) * mm + (p2 - p1) * (2.0 * mt) + (p3 - p2) * tt;
    return PointDeriv(p, q);
}

fn cubic_start_tangent(p0: vec2f, p1: vec2f, p2: vec2f, p3: vec2f) -> vec2f {
    let EPS = 1e-12;
    let d01 = p1 - p0;
    let d02 = p2 - p0;
    let d03 = p3 - p0;
    return select(select(d03, d02, dot(d02, d02) > EPS), d01, dot(d01, d01) > EPS);
}

fn cubic_end_tangent(p0: vec2f, p1: vec2f, p2: vec2f, p3: vec2f) -> vec2f {
    let EPS = 1e-12;
    let d23 = p3 - p2;
    let d13 = p3 - p1;
    let d03 = p3 - p0;
    return select(select(d03, d13, dot(d13, d13) > EPS), d23, dot(d23, d23) > EPS);
}

fn cubic_start_normal(p0: vec2f, p1: vec2f, p2: vec2f, p3: vec2f) -> vec2f {
    let tangent = normalize(cubic_start_tangent(p0, p1, p2, p3));
    return vec2(-tangent.y, tangent.x);
}

fn cubic_end_normal(p0: vec2f, p1: vec2f, p2: vec2f, p3: vec2f) -> vec2f {
    let tangent = normalize(cubic_end_tangent(p0, p1, p2, p3));
    return vec2(-tangent.y, tangent.x);
}

const ESPC_ROBUST_NORMAL = 0;
const ESPC_ROBUST_LOW_K1 = 1;
const ESPC_ROBUST_LOW_DIST = 2;

// This function flattens a cubic Bézier by first converting it into Euler spiral
// segments, and then computes a near-optimal flattening of the parallel curves of
// the Euler spiral segments.
fn flatten_euler(
    cubic: CubicPoints,
    path_ix: u32,
    local_to_device: Transform,
    offset: f32,
    start_p: vec2f,
    end_p: vec2f,
) {
    var p0: vec2f;
    var p1: vec2f;
    var p2: vec2f;
    var p3: vec2f;
    var scale: f32;
    var transform: Transform;
    var t_start = start_p;
    var t_end = end_p;
    if offset == 0. {
        let t = local_to_device;
        p0 = transform_apply(t, cubic.p0);
        p1 = transform_apply(t, cubic.p1);
        p2 = transform_apply(t, cubic.p2);
        p3 = transform_apply(t, cubic.p3);
        scale = 1.;
        transform = transform_identity();
        t_start = p0;
        t_end = p3;
    } else {
        p0 = cubic.p0;
        p1 = cubic.p1;
        p2 = cubic.p2;
        p3 = cubic.p3;

        transform = local_to_device;
        let mat = transform.mat;
        // The scale is the semi-major axis of the ellipse given by the unit
        // circle transformed by `transform`. This is the greater of the two
        // singular values of the 2x2 `transform` matrix (ignoring
        // translation).
        scale = 0.5 * (length(vec2(mat.x + mat.w, mat.y - mat.z)) +
                length(vec2(mat.x - mat.w, mat.y + mat.z)));
    }

    // Drop zero length lines. This is an exact equality test because dropping very short
    // line segments may result in loss of watertightness.
    if all(p0 == p1) && all(p0 == p2) && all(p0 == p3) {
        return;
    }

    let tol = 0.25;
    var t0_u = 0u;
    var dt = 1.0;
    var last_p = p0;
    var last_q = p1 - p0;
    if dot(last_q, last_q) < DERIV_THRESH_SQUARED {
        last_q = eval_cubic_and_deriv(p0, p1, p2, p3, DERIV_EPS).deriv;
    }
    var last_t = 0.0;
    var lp0 = t_start;
    loop {
        let t0 = f32(t0_u) * dt;
        if t0 == 1.0 {
            break;
        }
        var t1 = t0 + dt;
        let this_p0 = last_p;
        let this_q0 = last_q;
        var this_pq1 = eval_cubic_and_deriv(p0, p1, p2, p3, t1);
        if dot(this_pq1.deriv, this_pq1.deriv) < DERIV_THRESH_SQUARED {
            let new_pq1 = eval_cubic_and_deriv(p0, p1, p2, p3, t1 - DERIV_EPS);
            this_pq1.deriv = new_pq1.deriv;
            if t1 < 1.0 {
                this_pq1.point = new_pq1.point;
                t1 = t1 - DERIV_EPS;
            }
        }
        let actual_dt = t1 - last_t;
        let cubic_params = cubic_from_points_derivs(this_p0, this_pq1.point, this_q0, this_pq1.deriv, actual_dt);
        if cubic_params.err * scale <= tol || dt <= SUBDIV_LIMIT {
            let euler_params = es_params_from_angles(cubic_params.th0, cubic_params.th1);
            let es = es_seg_from_params(this_p0, this_pq1.point, euler_params);
            let k0 = es.params.k0 - 0.5 * es.params.k1;
            let k1 = es.params.k1;
            let normalized_offset = offset / cubic_params.chord_len;
            let dist_scaled = normalized_offset * es.params.ch;
            let scale_multiplier = sqrt(0.125 * scale * cubic_params.chord_len / (es.params.ch * tol));
            var a = 0.0;
            var b = 0.0;
            var integral = 0.0;
            var int0 = 0.0;
            var n_frac: f32;
            var robust = ESPC_ROBUST_NORMAL;
            if abs(k1) < K1_THRESH {
                let k = es.params.k0;
                n_frac = sqrt(abs(k * (k * dist_scaled + 1.0)));
                robust = ESPC_ROBUST_LOW_K1;
            } else if abs(dist_scaled) < DIST_THRESH {
                a = k1;
                b = k0;
                int0 = pow_1_5_signed(b);
                let int1 = pow_1_5_signed(a + b);
                integral = int1 - int0;
                n_frac = (2. / 3.) * integral / a;
                robust = ESPC_ROBUST_LOW_DIST;
            } else {
                a = -2.0 * dist_scaled * k1;
                b = -1.0 - 2.0 * dist_scaled * k0;
                int0 = espc_int_approx(b);
                let int1 = espc_int_approx(a + b);
                integral = int1 - int0;
                let k_peak = k0 - k1 * b / a;
                let integrand_peak = sqrt(abs(k_peak * (k_peak * dist_scaled + 1.0)));
                n_frac = integral * integrand_peak / a;
            }
            // Bound number of subdivisions to a reasonable number when the scale is huge.
            // This may give slightly incorrect rendering but avoids hangs.
            // TODO: aggressively cull to viewport
            let n = clamp(ceil(n_frac * scale_multiplier), 1.0, 100.0);
            for (var i = 0u; i < u32(n); i++) {
                var lp1: vec2f;
                if i + 1u == u32(n) && t1 == 1.0 {
                    lp1 = t_end;
                } else {
                    let t = f32(i + 1u) / n;
                    var s = t;
                    if robust != ESPC_ROBUST_LOW_K1 {
                        let u = integral * t + int0;
                        var inv: f32;
                        if robust == ESPC_ROBUST_LOW_DIST {
                            inv = pow(abs(u), 2. / 3.) * sign(u);
                        } else {
                            inv = espc_int_inv_approx(u);
                        }
                        s = (inv - b) / a;
                    }
                    lp1 = es_seg_eval_with_offset(es, s, normalized_offset);
                }
                let l0 = select(lp1, lp0, offset >= 0.);
                let l1 = select(lp0, lp1, offset >= 0.);
                output_line_with_transform(path_ix, l0, l1, transform);
                lp0 = lp1;
            }
            last_p = this_pq1.point;
            last_q = this_pq1.deriv;
            last_t = t1;
            t0_u += 1u;
            let shift = countTrailingZeros(t0_u);
            t0_u >>= shift;
            dt *= f32(1u << shift);
        } else {
            t0_u = t0_u * 2u;
            dt *= 0.5;
        }
    }
}

// Flattens the circular arc that subtends the angle begin-center-end. It is assumed that
// ||begin - center|| == ||end - center||. `begin`, `end`, and `center` are defined in the path's
// local coordinate space.
//
// The direction of the arc is always a counter-clockwise (Y-down) rotation starting from `begin`,
// towards `end`, centered at `center`, and will be subtended by `angle` (which is assumed to be
// positive). A line segment will always be drawn from the arc's terminus to `end`, regardless of
// `angle`.
//
// `begin`, `end`, center`, and `angle` should be chosen carefully to ensure a smooth arc with the
// correct winding.
fn flatten_arc(
    path_ix: u32, begin: vec2f, end: vec2f, center: vec2f, angle: f32, transform: Transform
) {
    var p0 = transform_apply(transform, begin);
    var r = begin - center;

    let MIN_THETA = 0.0001;
    let tol = 0.25;
    let radius = max(tol, length(p0 - transform_apply(transform, center)));
    let theta = max(MIN_THETA, 2. * acos(1. - tol / radius));

    // Always output at least one line so that we always draw the chord.
    let n_lines = max(1u, u32(ceil(angle / theta)));

    let c = cos(theta);
    let s = sin(theta);
    let rot = mat2x2(c, -s, s, c);

    let line_ix = atomicAdd(&bump.lines, n_lines);
    for (var i = 0u; i < n_lines - 1u; i += 1u) {
        r = rot * r;
        let p1 = transform_apply(transform, center + r);
        write_line(line_ix + i, path_ix, p0, p1);
        p0 = p1;
    }
    let p1 = transform_apply(transform, end);
    write_line(line_ix + n_lines - 1u, path_ix, p0, p1);
}

fn draw_cap(
    path_ix: u32, cap_style: u32, point: vec2f,
    cap0: vec2f, cap1: vec2f, offset_tangent: vec2f,
    transform: Transform,
) {
    if cap_style == STYLE_FLAGS_CAP_ROUND {
        flatten_arc(path_ix, cap0, cap1, point, 3.1415927, transform);
        return;
    }

    var start = cap0;
    var end = cap1;
    let is_square = (cap_style == STYLE_FLAGS_CAP_SQUARE);
    let line_ix = atomicAdd(&bump.lines, select(1u, 3u, is_square));
    if is_square {
        let v = offset_tangent;
        let p0 = start + v;
        let p1 = end + v;
        write_line_with_transform(line_ix + 1u, path_ix, start, p0, transform);
        write_line_with_transform(line_ix + 2u, path_ix, p1, end, transform);
        start = p0;
        end = p1;
    }
    write_line_with_transform(line_ix, path_ix, start, end, transform);
}

fn draw_join(
    path_ix: u32, style_flags: u32, p0: vec2f,
    tan_prev: vec2f, tan_next: vec2f,
    n_prev: vec2f, n_next: vec2f,
    transform: Transform,
) {
    var front0 = p0 + n_prev;
    let front1 = p0 + n_next;
    var back0 = p0 - n_next;
    let back1 = p0 - n_prev;

    let cr = tan_prev.x * tan_next.y - tan_prev.y * tan_next.x;
    let d = dot(tan_prev, tan_next);

    switch style_flags & STYLE_FLAGS_JOIN_MASK {
        case STYLE_FLAGS_JOIN_BEVEL: {
            output_two_lines_with_transform(path_ix, front0, front1, back0, back1, transform);
        }
        case STYLE_FLAGS_JOIN_MITER: {
            let hypot = length(vec2f(cr, d));
            let miter_limit = unpack2x16float(style_flags & STYLE_MITER_LIMIT_MASK)[0];

            var line_ix: u32;
            // Given the two tangents `tan_prev` and `tan_next` arranged tail-to-tail, the
            // miter length ratio is `1 / |cos(theta/2)|`, where `theta` is the angle
            // between the tangents.
            //
            // `hypot` is `|tan_prev| * |tan_next|` (since cr^2 + d^2 = |a|^2 |b|^2) and
            // `hypot + d` is `2 * |tan_prev| * |tan_next| * cos^2(theta/2)`. After
            // rearranging, the following tests whether `1/|cos(theta/2)| < miter_limit`.
            //
            // Also avoid the miter computation when `cr` is very small; the intersection
            // math divides by `cr` and becomes numerically unstable for near-collinear
            // tangents.
            if 2. * hypot < (hypot + d) * miter_limit * miter_limit
                && abs(cr) > TANGENT_THRESH * TANGENT_THRESH
            {
                let is_backside = cr > 0.;
                let fp_last = select(front0, back1, is_backside);
                let fp_this = select(front1, back0, is_backside);
                let p = select(front0, back0, is_backside);

                let v = fp_this - fp_last;
                let h = (tan_prev.x * v.y - tan_prev.y * v.x) / cr;
                let miter_pt = fp_this - tan_next * h;

                line_ix = atomicAdd(&bump.lines, 3u);
                write_line_with_transform(line_ix, path_ix, p, miter_pt, transform);
                line_ix += 1u;

                if is_backside {
                    back0 = miter_pt;
                } else {
                    front0 = miter_pt;
                }
            } else {
                line_ix = atomicAdd(&bump.lines, 2u);
            }
            write_line_with_transform(line_ix, path_ix, front0, front1, transform);
            write_line_with_transform(line_ix + 1u, path_ix, back0, back1, transform);
        }
        case STYLE_FLAGS_JOIN_ROUND: {
            var arc0: vec2f;
            var arc1: vec2f;
            var other0: vec2f;
            var other1: vec2f;
            if cr > 0. {
                arc0 = back0;
                arc1 = back1;
                other0 = front0;
                other1 = front1;
            } else {
                arc0 = front0;
                arc1 = front1;
                other0 = back0;
                other1 = back1;
            }
            flatten_arc(path_ix, arc0, arc1, p0, abs(atan2(cr, d)), transform);
            output_line_with_transform(path_ix, other0, other1, transform);
        }
        default: {}
    }
}

fn read_f32_point(ix: u32) -> vec2f {
    let x = bitcast<f32>(scene[pathdata_base + ix]);
    let y = bitcast<f32>(scene[pathdata_base + ix + 1u]);
    return vec2(x, y);
}

fn read_i16_point(ix: u32) -> vec2f {
    let raw = scene[pathdata_base + ix];
    let x = f32(i32(raw << 16u) >> 16u);
    let y = f32(i32(raw) >> 16u);
    return vec2(x, y);
}

struct Transform {
    mat: vec4f,
    translate: vec2f,
}

fn transform_identity() -> Transform {
    return Transform(vec4(1., 0., 0., 1.), vec2(0.));
}

fn read_transform(transform_base: u32, ix: u32) -> Transform {
    let base = transform_base + ix * 6u;
    let c0 = bitcast<f32>(scene[base]);
    let c1 = bitcast<f32>(scene[base + 1u]);
    let c2 = bitcast<f32>(scene[base + 2u]);
    let c3 = bitcast<f32>(scene[base + 3u]);
    let c4 = bitcast<f32>(scene[base + 4u]);
    let c5 = bitcast<f32>(scene[base + 5u]);
    let mat = vec4(c0, c1, c2, c3);
    let translate = vec2(c4, c5);
    return Transform(mat, translate);
}

fn transform_apply(transform: Transform, p: vec2f) -> vec2f {
    let px = fma(transform.mat.x, p.x, fma(transform.mat.z, p.y, transform.translate.x));
    let py = fma(transform.mat.y, p.x, fma(transform.mat.w, p.y, transform.translate.y));
    return vec2(px, py);
}

fn round_down(x: f32) -> i32 {
    return i32(floor(x));
}

fn round_up(x: f32) -> i32 {
    return i32(ceil(x));
}

struct PathTagData {
    tag_byte: u32,
    monoid: TagMonoid,
}

fn compute_tag_monoid(ix: u32) -> PathTagData {
    let tag_word = scene[config.pathtag_base + (ix >> 2u)];
    let shift = (ix & 3u) * 8u;
    var tm = reduce_tag(tag_word & ((1u << shift) - 1u));
    // TODO: this can be a read buf overflow. Conditionalize by tag byte?
    tm = combine_tag_monoid(tag_monoids[ix >> 2u], tm);
    var tag_byte = (tag_word >> shift) & 0xffu;
    // We no longer encode an initial transform and style so these
    // are off by one.
    // Note: an alternative would be to adjust config.transform_base and
    // config.style_base.
    tm.trans_ix -= 1u;
    tm.style_ix -= STYLE_SIZE_IN_WORDS;
    return PathTagData(tag_byte, tm);
}

struct CubicPoints {
    p0: vec2f,
    p1: vec2f,
    p2: vec2f,
    p3: vec2f,
}

fn read_path_segment(tag: PathTagData, is_stroke: bool) -> CubicPoints {
    var p0: vec2f;
    var p1: vec2f;
    var p2: vec2f;
    var p3: vec2f;

    var seg_type = tag.tag_byte & PATH_TAG_SEG_TYPE;
    let pathseg_offset = tag.monoid.pathseg_offset;
    let is_stroke_cap_marker = is_stroke && (tag.tag_byte & PATH_TAG_SUBPATH_END) != 0u;
    let is_open = seg_type == PATH_TAG_QUADTO;

    if (tag.tag_byte & PATH_TAG_F32) != 0u {
        p0 = read_f32_point(pathseg_offset);
        p1 = read_f32_point(pathseg_offset + 2u);
        if seg_type >= PATH_TAG_QUADTO {
            p2 = read_f32_point(pathseg_offset + 4u);
            if seg_type == PATH_TAG_CUBICTO {
                p3 = read_f32_point(pathseg_offset + 6u);
            }
        }
    } else {
        p0 = read_i16_point(pathseg_offset);
        p1 = read_i16_point(pathseg_offset + 1u);
        if seg_type >= PATH_TAG_QUADTO {
            p2 = read_i16_point(pathseg_offset + 2u);
            if seg_type == PATH_TAG_CUBICTO {
                p3 = read_i16_point(pathseg_offset + 3u);
            }
        }
    }

    if is_stroke_cap_marker && is_open {
        // The stroke cap marker for an open path is encoded as a quadto where the p1 and p2 store
        // the start control point of the subpath and together with p2 forms the start tangent. p0
        // is ignored.
        //
        // This is encoded this way because encoding this as a lineto would require adding a moveto,
        // which would terminate the subpath too early (by setting the SUBPATH_END on the
        // segment preceding the cap marker). This scheme is only used for strokes.
        p0 = p1;
        p1 = p2;
        seg_type = PATH_TAG_LINETO;
    }

    // Degree-raise
    if seg_type == PATH_TAG_LINETO {
        p3 = p1;
        p2 = p3 + (1.0 / 3.0) * (p0 - p3);
        p1 = p0 + (1.0 / 3.0) * (p3 - p0);
    } else if seg_type == PATH_TAG_QUADTO {
        p3 = p2;
        p2 = p1 + (1.0 / 3.0) * (p2 - p1);
        p1 = p1 + (1.0 / 3.0) * (p0 - p1);
    }

    return CubicPoints(p0, p1, p2, p3);
}

// Writes a line into a the `lines` buffer at a pre-allocated location designated by `line_ix`.
fn write_line(line_ix: u32, path_ix: u32, p0: vec2f, p1: vec2f) {
    bbox = vec4(min(bbox.xy, min(p0, p1)), max(bbox.zw, max(p0, p1)));
    if line_ix < config.lines_size {
        lines[line_ix] = LineSoup(path_ix, p0, p1);
    }
}

fn write_line_with_transform(line_ix: u32, path_ix: u32, p0: vec2f, p1: vec2f, t: Transform) {
    let tp0 = transform_apply(t, p0);
    let tp1 = transform_apply(t, p1);
    write_line(line_ix, path_ix, tp0, tp1);
}

fn output_line(path_ix: u32, p0: vec2f, p1: vec2f) {
    let line_ix = atomicAdd(&bump.lines, 1u);
    write_line(line_ix, path_ix, p0, p1);
}

fn output_line_with_transform(path_ix: u32, p0: vec2f, p1: vec2f, transform: Transform) {
    let line_ix = atomicAdd(&bump.lines, 1u);
    write_line_with_transform(line_ix, path_ix, p0, p1, transform);
}

fn output_two_lines_with_transform(
    path_ix: u32,
    p00: vec2f, p01: vec2f,
    p10: vec2f, p11: vec2f,
    transform: Transform
) {
    let line_ix = atomicAdd(&bump.lines, 2u);
    write_line_with_transform(line_ix, path_ix, p00, p01, transform);
    write_line_with_transform(line_ix + 1u, path_ix, p10, p11, transform);
}

struct NeighboringSegment {
    do_join: bool,

    // Device-space start tangent vector
    tangent: vec2f,
}

fn read_neighboring_segment(ix: u32) -> NeighboringSegment {
    let tag = compute_tag_monoid(ix);
    let pts = read_path_segment(tag, true);

    let is_closed = (tag.tag_byte & PATH_TAG_SEG_TYPE) == PATH_TAG_LINETO;
    let is_stroke_cap_marker = (tag.tag_byte & PATH_TAG_SUBPATH_END) != 0u;
    let do_join = !is_stroke_cap_marker || is_closed;
    var tangent = pts.p3 - pts.p0;
    if !is_stroke_cap_marker {
        tangent = cubic_start_tangent(pts.p0, pts.p1, pts.p2, pts.p3);
    }
    return NeighboringSegment(do_join, tangent);
}

// `pathdata_base` is decoded once and reused by helpers above.
var<private> pathdata_base: u32;

// This is the bounding box of the shape flattened by a single shader invocation. It gets modified
// during LineSoup generation.
var<private> bbox: vec4f;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let ix = global_id.x;
    pathdata_base = config.pathdata_base;
    bbox = vec4(1e31, 1e31, -1e31, -1e31);

    let tag = compute_tag_monoid(ix);
    let path_ix = tag.monoid.path_ix;
    let style_ix = tag.monoid.style_ix;
    let trans_ix = tag.monoid.trans_ix;

    let out = &path_bboxes[path_ix];
    let style_flags = scene[config.style_base + style_ix];
    // The fill bit is always set to 0 for strokes which represents a non-zero fill.
    let draw_flags = select(DRAW_INFO_FLAGS_FILL_RULE_BIT, 0u, (style_flags & STYLE_FLAGS_FILL) == 0u);
    if (tag.tag_byte & PATH_TAG_PATH) != 0u {
        (*out).draw_flags = draw_flags;
        (*out).trans_ix = trans_ix;
    }
    // Decode path data
    let seg_type = tag.tag_byte & PATH_TAG_SEG_TYPE;
    if seg_type != 0u {
        let is_stroke = (style_flags & STYLE_FLAGS_STYLE) != 0u;
        let transform = read_transform(config.transform_base, trans_ix);
        let pts = read_path_segment(tag, is_stroke);

        if is_stroke {
            let linewidth = bitcast<f32>(scene[config.style_base + style_ix + 1u]);
            let offset = 0.5 * linewidth;

            let is_open = (tag.tag_byte & PATH_TAG_SEG_TYPE) != PATH_TAG_LINETO;
            let is_stroke_cap_marker = (tag.tag_byte & PATH_TAG_SUBPATH_END) != 0u;
            if is_stroke_cap_marker {
                if is_open {
                    // Draw start cap
                    let tangent = pts.p3 - pts.p0;
                    let offset_tangent = offset * normalize(tangent);
                    let n = offset_tangent.yx * vec2f(-1., 1.);
                    draw_cap(path_ix, (style_flags & STYLE_FLAGS_START_CAP_MASK) >> 2u,
                             pts.p0, pts.p0 - n, pts.p0 + n, -offset_tangent, transform);
                } else {
                    // Don't draw anything if the path is closed.
                }
            } else {
                // Read the neighboring segment.
                let neighbor = read_neighboring_segment(ix + 1u);
                var tan_start = cubic_start_tangent(pts.p0, pts.p1, pts.p2, pts.p3);
                if dot(tan_start, tan_start) < TANGENT_THRESH * TANGENT_THRESH {
                    tan_start = vec2(TANGENT_THRESH, 0.);
                }
                var tan_prev = cubic_end_tangent(pts.p0, pts.p1, pts.p2, pts.p3);
                if dot(tan_prev, tan_prev) < TANGENT_THRESH * TANGENT_THRESH {
                    tan_prev = vec2(TANGENT_THRESH, 0.);
                }
                var tan_next = neighbor.tangent;
                if dot(tan_next, tan_next) < TANGENT_THRESH * TANGENT_THRESH {
                    tan_next = vec2(TANGENT_THRESH, 0.);
                }
                let n_start = offset * normalize(vec2(-tan_start.y, tan_start.x));
                let offset_tangent = offset * normalize(tan_prev);
                let n_prev = offset_tangent.yx * vec2f(-1., 1.);
                let n_next = offset * normalize(tan_next).yx * vec2f(-1., 1.);

                // Render offset curves
                flatten_euler(pts, path_ix, transform, offset, pts.p0 + n_start, pts.p3 + n_prev);
                flatten_euler(pts, path_ix, transform, -offset, pts.p0 - n_start, pts.p3 - n_prev);

                if neighbor.do_join {
                    draw_join(path_ix, style_flags, pts.p3, tan_prev, tan_next,
                              n_prev, n_next, transform);
                } else {
                    // Draw end cap.
                    draw_cap(path_ix, (style_flags & STYLE_FLAGS_END_CAP_MASK),
                             pts.p3, pts.p3 + n_prev, pts.p3 - n_prev, offset_tangent, transform);
                }
            }
        } else {
            let offset = 0.;
            flatten_euler(pts, path_ix, transform, offset, pts.p0, pts.p3);
        }
        // Update bounding box using atomics only. Computing a monoid is a
        // potential future optimization.
        if bbox.z > bbox.x || bbox.w > bbox.y {
            atomicMin(&(*out).x0, round_down(bbox.x));
            atomicMin(&(*out).y0, round_down(bbox.y));
            atomicMax(&(*out).x1, round_up(bbox.z));
            atomicMax(&(*out).y1, round_up(bbox.w));
        }
    }
}
