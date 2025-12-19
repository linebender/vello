// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

use std::f32::consts::FRAC_1_SQRT_2;

use super::{
    CpuBinding,
    euler::{
        CubicParams, EulerParams, EulerSeg, TANGENT_THRESH, espc_int_approx, espc_int_inv_approx,
    },
    util::{ROBUST_EPSILON, Transform, Vec2},
};
use vello_encoding::math::f16_to_f32;
use vello_encoding::{
    BumpAllocators, ConfigUniform, DRAW_INFO_FLAGS_FILL_RULE_BIT, LineSoup, Monoid, PathBbox,
    PathMonoid, PathTag, Style,
};

// TODO: remove this
macro_rules! log {
    ($($arg:tt)*) => {{
        //println!($($arg)*);
    }};
}

// Note to readers: this file contains sophisticated techniques for expanding stroke
// outlines to flattened filled outlines, based on Euler spirals as an intermediate
// curve representation. In some cases, there are explanatory comments in the
// corresponding `cpu_shaders/` files (`flatten.rs` and the supporting `euler.rs`).
// A paper is in the works explaining the techniques in more detail.

/// Threshold below which a derivative is considered too small.
const DERIV_THRESH: f32 = 1e-6;
/// Amount to nudge t when derivative is near-zero.
const DERIV_EPS: f32 = 1e-6;
// Limit for subdivision of cubic Béziers.
const SUBDIV_LIMIT: f32 = 1.0 / 65536.0;

/// Evaluate both the point and derivative of a cubic bezier.
fn eval_cubic_and_deriv(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: f32) -> (Vec2, Vec2) {
    let m = 1.0 - t;
    let mm = m * m;
    let mt = m * t;
    let tt = t * t;
    let p = p0 * (mm * m) + (p1 * (3.0 * mm) + p2 * (3.0 * mt) + p3 * tt) * t;
    let q = (p1 - p0) * mm + (p2 - p1) * (2.0 * mt) + (p3 - p2) * tt;
    (p, q)
}

fn cubic_start_tangent(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2) -> Vec2 {
    let d01 = p1 - p0;
    let d02 = p2 - p0;
    let d03 = p3 - p0;
    if d01.length_squared() > ROBUST_EPSILON {
        d01
    } else if d02.length_squared() > ROBUST_EPSILON {
        d02
    } else {
        d03
    }
}

fn cubic_end_tangent(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2) -> Vec2 {
    let d23 = p3 - p2;
    let d13 = p3 - p1;
    let d03 = p3 - p0;
    if d23.length_squared() > ROBUST_EPSILON {
        d23
    } else if d13.length_squared() > ROBUST_EPSILON {
        d13
    } else {
        d03
    }
}

fn write_line(
    line_ix: usize,
    path_ix: u32,
    p0: Vec2,
    p1: Vec2,
    bbox: &mut IntBbox,
    lines: &mut [LineSoup],
) {
    assert!(
        !p0.is_nan() && !p1.is_nan(),
        "wrote NaNs: p0: {p0:?}, p1: {p1:?}"
    );
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
    const MIN_THETA: f32 = 0.0001;

    let mut p0 = transform.apply(begin);
    let mut r = begin - center;
    let tol: f32 = 0.25;
    let radius = tol.max((p0 - transform.apply(center)).length());
    let theta = (2. * (1. - tol / radius).acos()).max(MIN_THETA);

    // Always output at least one line so that we always draw the chord.
    let n_lines = ((angle / theta).ceil() as u32).max(1);

    let (s, c) = theta.sin_cos();
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

/// A robustness strategy for the ESPC integral
enum EspcRobust {
    /// Both k1 and dist are large enough to divide by robustly.
    Normal,
    /// k1 is low, so model curve as a circular arc.
    LowK1,
    /// dist is low, so model curve as just an Euler spiral.
    LowDist,
}

fn flatten_euler(
    cubic: &CubicPoints,
    path_ix: u32,
    local_to_device: &Transform,
    offset: f32,
    start_p: Vec2,
    end_p: Vec2,
    line_ix: &mut usize,
    lines: &mut [LineSoup],
    bbox: &mut IntBbox,
) {
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
        let scale = 0.5
            * (Vec2::new(t[0] + t[3], t[1] - t[2]).length()
                + Vec2::new(t[0] - t[3], t[1] + t[2]).length());
        (
            cubic.p0,
            cubic.p1,
            cubic.p2,
            cubic.p3,
            scale,
            local_to_device.clone(),
        )
    };
    let (t_start, t_end) = if offset == 0.0 {
        (p0, p3)
    } else {
        (start_p, end_p)
    };

    // Drop zero length lines. This is an exact equality test because dropping very short
    // line segments may result in loss of watertightness. The parallel curves of zero
    // length lines add nothing to stroke outlines, but we still may need to draw caps.
    if p0 == p1 && p0 == p2 && p0 == p3 {
        return;
    }

    let tol: f32 = 0.25;
    let mut t0_u: u32 = 0;
    let mut dt: f32 = 1.;
    let mut last_p = p0;
    let mut last_q = p1 - p0;
    // We want to avoid near zero derivatives, so the general technique is to
    // detect, then sample a nearby t value if it fails to meet the threshold.
    if last_q.length_squared() < DERIV_THRESH.powi(2) {
        last_q = eval_cubic_and_deriv(p0, p1, p2, p3, DERIV_EPS).1;
    }
    let mut last_t = 0.;
    let mut lp0 = t_start;

    loop {
        let t0 = (t0_u as f32) * dt;
        if t0 == 1. {
            break;
        }
        log!("@@@ loop start: t0: {t0}, dt: {dt}");
        let mut t1 = t0 + dt;
        let this_p0 = last_p;
        let this_q0 = last_q;
        let (mut this_p1, mut this_q1) = eval_cubic_and_deriv(p0, p1, p2, p3, t1);
        if this_q1.length_squared() < DERIV_THRESH.powi(2) {
            let (new_p1, new_q1) = eval_cubic_and_deriv(p0, p1, p2, p3, t1 - DERIV_EPS);
            this_q1 = new_q1;
            // Change just the derivative at the endpoint, but also move the point so it
            // matches the derivative exactly if in the interior.
            if t1 < 1. {
                this_p1 = new_p1;
                t1 -= DERIV_EPS;
            }
        }
        let actual_dt = t1 - last_t;
        let cubic_params =
            CubicParams::from_points_derivs(this_p0, this_p1, this_q0, this_q1, actual_dt);
        log!(
            "@@@   loop: p0={this_p0:?} p1={this_p1:?} q0={this_q0:?} q1={this_q1:?} {cubic_params:?} t0: {t0}, t1: {t1}, dt: {dt}"
        );
        if cubic_params.err * scale <= tol || dt <= SUBDIV_LIMIT {
            log!("@@@   error within tolerance");
            let euler_params = EulerParams::from_angles(cubic_params.th0, cubic_params.th1);
            let es = EulerSeg::from_params(this_p0, this_p1, euler_params);

            let (k0, k1) = (es.params.k0 - 0.5 * es.params.k1, es.params.k1);

            // compute forward integral to determine number of subdivisions
            let normalized_offset = offset / cubic_params.chord_len;
            let dist_scaled = normalized_offset * es.params.ch;
            // The number of subdivisions for curvature = 1
            let scale_multiplier = 0.5
                * FRAC_1_SQRT_2
                * (scale * cubic_params.chord_len / (es.params.ch * tol)).sqrt();
            // TODO: tune these thresholds
            const K1_THRESH: f32 = 1e-3;
            const DIST_THRESH: f32 = 1e-3;
            let mut a = 0.0;
            let mut b = 0.0;
            let mut integral = 0.0;
            let mut int0 = 0.0;
            let (n_frac, robust) = if k1.abs() < K1_THRESH {
                let k = k0 + 0.5 * k1;
                let n_frac = (k * (k * dist_scaled + 1.0)).abs().sqrt();
                (n_frac, EspcRobust::LowK1)
            } else if dist_scaled.abs() < DIST_THRESH {
                let f = |x: f32| x * x.abs().sqrt();
                a = k1;
                b = k0;
                int0 = f(b);
                let int1 = f(a + b);
                integral = int1 - int0;
                //println!("int0={int0}, int1={int1} a={a} b={b}");
                let n_frac = (2. / 3.) * integral / a;
                (n_frac, EspcRobust::LowDist)
            } else {
                a = -2.0 * dist_scaled * k1;
                b = -1.0 - 2.0 * dist_scaled * k0;
                int0 = espc_int_approx(b);
                let int1 = espc_int_approx(a + b);
                integral = int1 - int0;
                let k_peak = k0 - k1 * b / a;
                let integrand_peak = (k_peak * (k_peak * dist_scaled + 1.0)).abs().sqrt();
                let scaled_int = integral * integrand_peak / a;
                let n_frac = scaled_int;
                (n_frac, EspcRobust::Normal)
            };
            let n = (n_frac * scale_multiplier).ceil().clamp(1.0, 100.0);

            // Flatten line segments
            log!("@@@   loop: lines: {n}");
            assert!(!n.is_nan());
            for i in 0..n as usize {
                let lp1 = if i == n as usize - 1 && t1 == 1.0 {
                    t_end
                } else {
                    let t = (i + 1) as f32 / n;
                    let s = match robust {
                        EspcRobust::LowK1 => t,
                        // Note opportunities to minimize divergence
                        EspcRobust::LowDist => {
                            let c = (integral * t + int0).cbrt();
                            let inv = c * c.abs();
                            (inv - b) / a
                        }
                        EspcRobust::Normal => {
                            let inv = espc_int_inv_approx(integral * t + int0);
                            (inv - b) / a
                        }
                    };
                    es.eval_with_offset(s, normalized_offset)
                };
                let l0 = if offset >= 0. { lp0 } else { lp1 };
                let l1 = if offset >= 0. { lp1 } else { lp0 };
                output_line_with_transform(path_ix, l0, l1, &transform, line_ix, lines, bbox);
                lp0 = lp1;
            }
            last_p = this_p1;
            last_q = this_q1;
            last_t = t1;
            // Advance segment to next range. Beginning of segment is the end of
            // this one. The number of trailing zeros represents the number of stack
            // frames to pop in the recursive version of adaptive subdivision, and
            // each stack pop represents doubling of the size of the range.
            t0_u += 1;
            let shift = t0_u.trailing_zeros();
            t0_u >>= shift;
            dt *= (1 << shift) as f32;
        } else {
            // Subdivide; halve the size of the range while retaining its start.
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
            if front0 != front1 && back0 != back1 {
                output_two_lines_with_transform(
                    path_ix, front0, front1, back0, back1, transform, line_ix, lines, bbox,
                );
            }
        }
        Style::FLAGS_JOIN_BITS_MITER => {
            let hypot = cr.hypot(d);
            let miter_limit = f16_to_f32((style_flags & Style::MITER_LIMIT_MASK) as u16);

            if 2. * hypot < (hypot + d) * miter_limit * miter_limit
                && cr.abs() > TANGENT_THRESH.powi(2)
            {
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
        Self {
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
    // We no longer encode an initial transform and style so these
    // are off by one.
    // We wrap here because these values will return to positive values later
    // (when we add style_base)
    tm.trans_ix = tm.trans_ix.wrapping_sub(1);
    tm.style_ix = tm.style_ix.wrapping_sub(size_of::<Style>() as u32 / 4);
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
    let tangent = if is_stroke_cap_marker {
        pts.p3 - pts.p0
    } else {
        cubic_start_tangent(pts.p0, pts.p1, pts.p2, pts.p3)
    };
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
        let style_flags = scene[(config.layout.style_base.wrapping_add(style_ix)) as usize];
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
                        let tangent = pts.p3 - pts.p0;
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
                    // Read the neighboring segment.
                    let neighbor =
                        read_neighboring_segment(ix + 1, pathtags, pathdata, tag_monoids);
                    let tan_prev = cubic_end_tangent(pts.p0, pts.p1, pts.p2, pts.p3);
                    let tan_next = neighbor.tangent;
                    let tan_start = cubic_start_tangent(pts.p0, pts.p1, pts.p2, pts.p3);
                    // TODO: be consistent w/ robustness here

                    // TODO: add NaN assertions to CPU shaders PR (when writing lines)
                    // TODO: not all zero-length segments are getting filtered out
                    // TODO: this is a hack. How to handle caps on degenerate stroke?
                    // TODO: debug tricky stroke by isolation
                    let tan_start = if tan_start.length_squared() < TANGENT_THRESH.powi(2) {
                        Vec2::new(TANGENT_THRESH, 0.)
                    } else {
                        tan_start
                    };
                    let tan_prev = if tan_prev.length_squared() < TANGENT_THRESH.powi(2) {
                        Vec2::new(TANGENT_THRESH, 0.)
                    } else {
                        tan_prev
                    };
                    let tan_next = if tan_next.length_squared() < TANGENT_THRESH.powi(2) {
                        Vec2::new(TANGENT_THRESH, 0.)
                    } else {
                        tan_next
                    };

                    let n_start = offset * Vec2::new(-tan_start.y, tan_start.x).normalize();
                    let offset_tangent = offset * tan_prev.normalize();
                    let n_prev = Vec2::new(-offset_tangent.y, offset_tangent.x);
                    let tan_next_norm = tan_next.normalize();
                    let n_next = offset * Vec2::new(-tan_next_norm.y, tan_next_norm.x);
                    log!("@ tan_prev: {:#?}", tan_prev);
                    log!("@ tan_next: {:#?}", tan_next);

                    // Render offset curves
                    flatten_euler(
                        &pts,
                        path_ix,
                        &transform,
                        offset,
                        pts.p0 + n_start,
                        pts.p3 + n_prev,
                        &mut line_ix,
                        lines,
                        &mut bbox,
                    );
                    flatten_euler(
                        &pts,
                        path_ix,
                        &transform,
                        -offset,
                        pts.p0 - n_start,
                        pts.p3 - n_prev,
                        &mut line_ix,
                        lines,
                        &mut bbox,
                    );

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
                flatten_euler(
                    &pts,
                    path_ix,
                    &transform,
                    /*offset*/ 0.,
                    pts.p0,
                    pts.p3,
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

pub fn flatten(n_wg: u32, resources: &[CpuBinding<'_>]) {
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
