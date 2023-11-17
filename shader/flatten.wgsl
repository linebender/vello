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

let D = 0.67;
fn approx_parabola_integral(x: f32) -> f32 {
    return x * inverseSqrt(sqrt(1.0 - D + (D * D * D * D + 0.25 * x * x)));
}

let B = 0.39;
fn approx_parabola_inv_integral(x: f32) -> f32 {
    return x * sqrt(1.0 - B + (B * B + 0.5 * x * x));
}

// Notes on fractional subdivision:
// --------------------------------
// The core of the existing flattening algorithm (see `flatten_cubic` below) is to approximate the
// original cubic Bézier into a simpler curve (quadratic Bézier), subdivided to meet the error
// bound, then apply flattening to that. Doing this the simplest way would put a subdivision point
// in the output at each subdivision point here. That in general does not match where the
// subdivision points would go in an optimal flattening. Fractional subdivision addresses that
// problem.
//
// The return value of this function (`val`) represents this fractional subdivision count and has
// the following meaning: an optimal subdivision of the quadratic into `val / 2` subdivisions
// will have an error `sqrt_tol^2` (i.e. the desired tolerance).
//
// In the non-cusp case, the error scales as the inverse square of `val` (doubling `val` causes the
// error to be one fourth), so the tolerance is actually not needed for the calculation (and gets
// applied in the caller). In the cusp case, this scaling breaks down and the tolerance parameter
// is needed to compute the correct result.
fn estimate_subdiv(p0: vec2f, p1: vec2f, p2: vec2f, sqrt_tol: f32) -> SubdivResult {
    let d01 = p1 - p0;
    let d12 = p2 - p1;
    let dd = d01 - d12;
    let cross = (p2.x - p0.x) * dd.y - (p2.y - p0.y) * dd.x;
    let cross_inv = select(1.0 / cross, 1.0e9, abs(cross) < 1.0e-9);
    let x0 = dot(d01, dd) * cross_inv;
    let x2 = dot(d12, dd) * cross_inv;
    let scale = abs(cross / (length(dd) * (x2 - x0)));

    let a0 = approx_parabola_integral(x0);
    let a2 = approx_parabola_integral(x2);
    var val = 0.0;
    if scale < 1e9 {
        let da = abs(a2 - a0);
        let sqrt_scale = sqrt(scale);
        if sign(x0) == sign(x2) {
            val = sqrt_scale;
        } else {
            let xmin = sqrt_tol / sqrt_scale;
            val = sqrt_tol / approx_parabola_integral(xmin);
        }
        val *= da;
    }
    return SubdivResult(val, a0, a2);
}

fn eval_quad(p0: vec2f, p1: vec2f, p2: vec2f, t: f32) -> vec2f {
    let mt = 1.0 - t;
    return p0 * (mt * mt) + (p1 * (mt * 2.0) + p2 * t) * t;
}

fn eval_cubic(p0: vec2f, p1: vec2f, p2: vec2f, p3: vec2f, t: f32) -> vec2f {
    let mt = 1.0 - t;
    return p0 * (mt * mt * mt) + (p1 * (mt * mt * 3.0) + (p2 * (mt * 3.0) + p3 * t) * t) * t;
}

fn eval_quad_tangent(p0: vec2f, p1: vec2f, p2: vec2f, t: f32) -> vec2f {
    let dp0 = 2. * (p1 - p0);
    let dp1 = 2. * (p2 - p1);
    return mix(dp0, dp1, t);
}

fn eval_quad_normal(p0: vec2f, p1: vec2f, p2: vec2f, t: f32) -> vec2f {
    let tangent = normalize(eval_quad_tangent(p0, p1, p2, t));
    return vec2(-tangent.y, tangent.x);
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

let MAX_QUADS = 16u;

// This function flattens a cubic Bézier by first converting it into quadratics and
// approximates the optimal flattening of those using a variation of the method described in
// https://raphlinus.github.io/graphics/curves/2019/12/23/flatten-quadbez.html.
//
// When the `offset` parameter is zero (i.e. the path is a "fill"), the flattening is performed
// directly on the transformed (device-space) control points as this produces near-optimal
// flattening even in the presence of a non-angle-preserving transform.
//
// When the `offset` is non-zero, the flattening is performed in the curve's local coordinate space
// and the offset curve gets transformed to device-space post-flattening. This handles
// non-angle-preserving transforms well while keeping the logic simple.
//
// When subdividing the cubic in its local coordinate space, the scale factor gets decomposed out of
// the local-to-device transform and gets factored into the tolerance threshold when estimating
// subdivisions.
fn flatten_cubic(cubic: CubicPoints, path_ix: u32, local_to_device: Transform, offset: f32) {
    var p0: vec2f;
    var p1: vec2f;
    var p2: vec2f;
    var p3: vec2f;
    var scale: f32;
    var transform: Transform;
    if offset == 0. {
        let t = local_to_device;
        p0 = transform_apply(t, cubic.p0);
        p1 = transform_apply(t, cubic.p1);
        p2 = transform_apply(t, cubic.p2);
        p3 = transform_apply(t, cubic.p3);
        scale = 1.;
        transform = transform_identity();
    } else {
        p0 = cubic.p0;
        p1 = cubic.p1;
        p2 = cubic.p2;
        p3 = cubic.p3;

        transform = local_to_device;
        let mat = transform.mat;
        scale = 0.5 * length(vec2(mat.x + mat.w, mat.y - mat.z)) +
                length(vec2(mat.x - mat.w, mat.y + mat.z));
    }

    let err_v = 3.0 * (p2 - p1) + p0 - p3;
    let err = dot(err_v, err_v);
    let ACCURACY = 0.25;
    let Q_ACCURACY = ACCURACY * 0.1;
    let REM_ACCURACY = ACCURACY - Q_ACCURACY;
    let MAX_HYPOT2 = 432.0 * Q_ACCURACY * Q_ACCURACY;
    let scaled_sqrt_tol = sqrt(REM_ACCURACY / scale);
    // Fudge the subdivision count metric to account for `scale` when the subdivision is done in local
    // coordinates.
    var n_quads = max(u32(ceil(pow(err * (1.0 / MAX_HYPOT2), 1.0 / 6.0)) * scale), 1u);
    n_quads = min(n_quads, MAX_QUADS);

    var keep_params: array<SubdivResult, MAX_QUADS>;
    var val = 0.0;
    var qp0 = p0;
    let step = 1.0 / f32(n_quads);
    for (var i = 0u; i < n_quads; i += 1u) {
        let t = f32(i + 1u) * step;
        let qp2 = eval_cubic(p0, p1, p2, p3, t);
        var qp1 = eval_cubic(p0, p1, p2, p3, t - 0.5 * step);
        qp1 = 2.0 * qp1 - 0.5 * (qp0 + qp2);

        // TODO: Estimate an accurate subdivision count for strokes
        let params = estimate_subdiv(qp0, qp1, qp2, scaled_sqrt_tol);
        keep_params[i] = params;
        val += params.val;
        qp0 = qp2;
    }

    // Normal vector to calculate the start point of the offset curve.
    var n0 = offset * cubic_start_normal(p0, p1, p2, p3);

    let n = max(u32(ceil(val * (0.5 / scaled_sqrt_tol))), 1u);
    var lp0 = p0;
    qp0 = p0;
    let v_step = val / f32(n);
    var n_out = 1u;
    var val_sum = 0.0;
    for (var i = 0u; i < n_quads; i += 1u) {
        let t = f32(i + 1u) * step;
        let qp2 = eval_cubic(p0, p1, p2, p3, t);
        var qp1 = eval_cubic(p0, p1, p2, p3, t - 0.5 * step);
        qp1 = 2.0 * qp1 - 0.5 * (qp0 + qp2);
        let params = keep_params[i];
        let u0 = approx_parabola_inv_integral(params.a0);
        let u2 = approx_parabola_inv_integral(params.a2);
        let uscale = 1.0 / (u2 - u0);
        var val_target = f32(n_out) * v_step;
        while n_out == n || val_target < val_sum + params.val {
            var lp1: vec2f;
            var t1: f32;
            if n_out == n {
                lp1 = p3;
                t1 = 1.;
            } else {
                let u = (val_target - val_sum) / params.val;
                let a = mix(params.a0, params.a2, u);
                let au = approx_parabola_inv_integral(a);
                let t = (au - u0) * uscale;
                t1 = t;
                lp1 = eval_quad(qp0, qp1, qp2, t);
            }

            // TODO: Instead of outputting two offset segments here, restructure this function as
            // "flatten_cubic_at_offset" such that it outputs one cubic at an offset. That should
            // more closely resemble the end state of this shader which will work like a state
            // machine.
            if offset > 0. {
                var n1: vec2f;
                if all(lp1 == p3) {
                    n1 = cubic_end_normal(p0, p1, p2, p3);
                } else {
                    n1 = eval_quad_normal(qp0, qp1, qp2, t1);
                }
                n1 *= offset;
                output_two_lines_with_transform(path_ix,
                                                lp0 + n0, lp1 + n1,
                                                lp1 - n1, lp0 - n0,
                                                transform);
                n0 = n1;
            } else {
                // Output line segment lp0..lp1
                output_line_with_transform(path_ix, lp0, lp1, transform);
            }
            n_out += 1u;
            val_target += v_step;
            lp0 = lp1;
        }
        val_sum += params.val;
        qp0 = qp2;
    }
}

// Flattens the circular arc that subtends the angle begin-center-end. It is assumed that
// ||begin - center|| == ||end - center||. `begin`, `end`, and `center` are defined in the path's
// local coordinate space.
fn flatten_arc(
    path_ix: u32, begin: vec2f, end: vec2f, center: vec2f, angle: f32, transform: Transform
) {
    var p0 = transform_apply(transform, begin);
    var r = begin - center;

    let EPS = 1e-9;
    let tol = 0.5;
    let radius = max(tol, length(p0 - transform_apply(transform, center)));
    let x = 1. - tol / radius;
    let theta = acos(clamp(2. * x * x - 1., -1., 1.));
    let MAX_LINES = 1000u;
    let n_lines = select(min(MAX_LINES, u32(ceil(6.2831853 / theta))), MAX_LINES, theta <= EPS);

    let th = angle / f32(n_lines);
    let c = cos(th);
    let s = sin(th);
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
        case /*STYLE_FLAGS_JOIN_BEVEL*/0u: {
            output_two_lines_with_transform(path_ix, front0, front1, back0, back1, transform);
        }
        case /*STYLE_FLAGS_JOIN_MITER*/0x10000000u: {
            let hypot = length(vec2f(cr, d));
            let miter_limit = unpack2x16float(style_flags & STYLE_MITER_LIMIT_MASK)[0];

            var line_ix: u32;
            if 2. * hypot < (hypot + d) * miter_limit * miter_limit && cr != 0. {
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
        case /*STYLE_FLAGS_JOIN_ROUND*/0x20000000u: {
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
    return transform.mat.xy * p.x + transform.mat.zw * p.y + transform.translate;
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
        p2 = mix(p3, p0, 1.0 / 3.0);
        p1 = mix(p0, p3, 1.0 / 3.0);
    } else if seg_type == PATH_TAG_QUADTO {
        p3 = p2;
        p2 = mix(p1, p2, 1.0 / 3.0);
        p1 = mix(p1, p0, 1.0 / 3.0);
    }

    return CubicPoints(p0, p1, p2, p3);
}

// Writes a line into a the `lines` buffer at a pre-allocated location designated by `line_ix`.
fn write_line(line_ix: u32, path_ix: u32, p0: vec2f, p1: vec2f) {
    bbox = vec4(min(bbox.xy, min(p0, p1)), max(bbox.zw, max(p0, p1)));
    lines[line_ix] = LineSoup(path_ix, p0, p1);
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
    p0: vec2f,

    // Device-space start tangent vector
    tangent: vec2f,
}

fn read_neighboring_segment(ix: u32) -> NeighboringSegment {
    let tag = compute_tag_monoid(ix);
    let pts = read_path_segment(tag, true);

    let is_closed = (tag.tag_byte & PATH_TAG_SEG_TYPE) == PATH_TAG_LINETO;
    let is_stroke_cap_marker = (tag.tag_byte & PATH_TAG_SUBPATH_END) != 0u;
    let do_join = !is_stroke_cap_marker || is_closed;
    let p0 = pts.p0;
    let tangent = cubic_start_tangent(pts.p0, pts.p1, pts.p2, pts.p3);
    return NeighboringSegment(do_join, p0, tangent);
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
                    let tangent = cubic_start_tangent(pts.p0, pts.p1, pts.p2, pts.p3);
                    let offset_tangent = offset * normalize(tangent);
                    let n = offset_tangent.yx * vec2f(-1., 1.);
                    draw_cap(path_ix, (style_flags & STYLE_FLAGS_START_CAP_MASK) >> 2u,
                             pts.p0, pts.p0 - n, pts.p0 + n, -offset_tangent, transform);
                } else {
                    // Don't draw anything if the path is closed.
                }
            } else {
                // Render offset curves
                flatten_cubic(pts, path_ix, transform, offset);

                // Read the neighboring segment.
                let neighbor = read_neighboring_segment(ix + 1u);
                let tan_prev = cubic_end_tangent(pts.p0, pts.p1, pts.p2, pts.p3);
                let tan_next = neighbor.tangent;
                let offset_tangent = offset * normalize(tan_prev);
                let n_prev = offset_tangent.yx * vec2f(-1., 1.);
                let n_next = offset * normalize(tan_next).yx * vec2f(-1., 1.);
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
            flatten_cubic(pts, path_ix, transform, /*offset*/ 0.);
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
