// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Flatten curves to lines

#import config
#import pathtag
#import segment
#import cubic
#import bump

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> scene: array<u32>;

@group(0) @binding(2)
var<storage> cubics: array<Cubic>;

@group(0) @binding(3)
var<storage, read_write> bump: BumpAllocators;

@group(0) @binding(4)
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

fn estimate_subdiv(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, sqrt_tol: f32) -> SubdivResult {
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

fn eval_quad(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, t: f32) -> vec2<f32> {
    let mt = 1.0 - t;
    return p0 * (mt * mt) + (p1 * (mt * 2.0) + p2 * t) * t;
}

fn eval_cubic(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, t: f32) -> vec2<f32> {
    let mt = 1.0 - t;
    return p0 * (mt * mt * mt) + (p1 * (mt * mt * 3.0) + (p2 * (mt * 3.0) + p3 * t) * t) * t;
}

let MAX_QUADS = 16u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let ix = global_id.x;
    let tag_word = scene[config.pathtag_base + (ix >> 2u)];
    let shift = (ix & 3u) * 8u;
    var tag_byte = (tag_word >> shift) & 0xffu;

    if (tag_byte & PATH_TAG_SEG_TYPE) != 0u {
        let cubic = cubics[global_id.x];
        let p0 = cubic.p0;
        let p1 = cubic.p1;
        let p2 = cubic.p2;
        let p3 = cubic.p3;
        let err_v = 3.0 * (p2 - p1) + p0 - p3;
        let err = dot(err_v, err_v);
        let ACCURACY = 0.25;
        let Q_ACCURACY = ACCURACY * 0.1;
        let REM_ACCURACY = (ACCURACY - Q_ACCURACY);
        let MAX_HYPOT2 = 432.0 * Q_ACCURACY * Q_ACCURACY;
        var n_quads = max(u32(ceil(pow(err * (1.0 / MAX_HYPOT2), 1.0 / 6.0))), 1u);
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
            let params = estimate_subdiv(qp0, qp1, qp2, sqrt(REM_ACCURACY));
            keep_params[i] = params;
            val += params.val;
            qp0 = qp2;
        }
        let n = max(u32(ceil(val * (0.5 / sqrt(REM_ACCURACY)))), 1u);
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
                var lp1: vec2<f32>;
                if n_out == n {
                    lp1 = p3;
                } else {
                    let u = (val_target - val_sum) / params.val;
                    let a = mix(params.a0, params.a2, u);
                    let au = approx_parabola_inv_integral(a);
                    let t = (au - u0) * uscale;
                    lp1 = eval_quad(qp0, qp1, qp2, t);
                }

                // Output line segment lp0..lp1
                let line_ix = atomicAdd(&bump.lines, 1u);
                // TODO: check failure
                lines[line_ix] = LineSoup(cubic.path_ix, lp0, lp1);
                n_out += 1u;
                val_target += v_step;
                lp0 = lp1;
            }
            val_sum += params.val;
            qp0 = qp2;
        }
    }
}
