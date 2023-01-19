// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Path coarse rasterization for the full implementation.

#import config
#import pathtag
#import tile
#import segment
#import cubic
#import bump

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> scene: array<u32>;

@group(0) @binding(2)
var<storage> tag_monoids: array<TagMonoid>;

@group(0) @binding(3)
var<storage> cubics: array<Cubic>;

@group(0) @binding(4)
var<storage> paths: array<Path>;

// We don't get this from import as it's the atomic version
struct AtomicTile {
    backdrop: atomic<i32>,
    segments: atomic<u32>,
}

@group(0) @binding(5)
var<storage, read_write> bump: BumpAllocators;

@group(0) @binding(6)
var<storage, read_write> tiles: array<AtomicTile>;

@group(0) @binding(7)
var<storage, read_write> segments: array<Segment>;

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
    let cross_inv = 1.0 / cross;
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

fn alloc_segment() -> u32 {
    var offset = atomicAdd(&bump.segments, 1u) + 1u;
    if offset + 1u > config.segments_size {
        offset = 0u;
        atomicOr(&bump.failed, STAGE_PATH_COARSE);
    }
    return offset;
}

let MAX_QUADS = 16u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    // Exit early if prior stages failed, as we can't run this stage.
    // We need to check only prior stages, as if this stage has failed in another workgroup, 
    // we still want to know this workgroup's memory requirement.   
    if (atomicLoad(&bump.failed) & (STAGE_BINNING | STAGE_TILE_ALLOC)) != 0u {
        return;
    }
    let ix = global_id.x;
    let tag_word = scene[config.pathtag_base + (ix >> 2u)];
    let shift = (ix & 3u) * 8u;
    var tag_byte = (tag_word >> shift) & 0xffu;

    if (tag_byte & PATH_TAG_SEG_TYPE) != 0u {
        // Discussion question: it might actually be cheaper to do the path segment
        // decoding & transform again rather than store the result in a buffer;
        // classic memory vs ALU tradeoff.
        let cubic = cubics[global_id.x];
        let path = paths[cubic.path_ix];
        let is_stroke = (cubic.flags & CUBIC_IS_STROKE) != 0u;
        let bbox = vec4<i32>(path.bbox);
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
                let xymin = min(lp0, lp1) - cubic.stroke;
                let xymax = max(lp0, lp1) + cubic.stroke;
                let dp = lp1 - lp0;
                let recip_dx = 1.0 / dp.x;
                let invslope = select(dp.x / dp.y, 1.0e9, abs(dp.y) < 1.0e-9);
                let SX = 1.0 / f32(TILE_WIDTH);
                let SY = 1.0 / f32(TILE_HEIGHT);
                let c = (cubic.stroke.x + abs(invslope) * (0.5 * f32(TILE_HEIGHT) + cubic.stroke.y)) * SX;
                let b = invslope;
                let a = (lp0.x - (lp0.y - 0.5 * f32(TILE_HEIGHT)) * b) * SX;
                var x0 = i32(floor(xymin.x * SX));
                var x1 = i32(floor(xymax.x * SX) + 1.0);
                var y0 = i32(floor(xymin.y * SY));
                var y1 = i32(floor(xymax.y * SY) + 1.0);
                x0 = clamp(x0, bbox.x, bbox.z);
                x1 = clamp(x1, bbox.x, bbox.z);
                y0 = clamp(y0, bbox.y, bbox.w);
                y1 = clamp(y1, bbox.y, bbox.w);
                var xc = a + b * f32(y0);
                let stride = bbox.z - bbox.x;
                var base = i32(path.tiles) + (y0 - bbox.y) * stride - bbox.x;
                var xray = i32(floor(lp0.x * SX));
                var last_xray = i32(floor(lp1.x * SX));
                if dp.y < 0.0 {
                    let tmp = xray;
                    xray = last_xray;
                    last_xray = tmp;
                }
                for (var y = y0; y < y1; y += 1) {
                    let tile_y0 = f32(y) * f32(TILE_HEIGHT);
                    let xbackdrop = max(xray + 1, bbox.x);
                    if !is_stroke && xymin.y < tile_y0 && xbackdrop < bbox.z {
                        let backdrop = select(-1, 1, dp.y < 0.0);
                        let tile_ix = base + xbackdrop;
                        atomicAdd(&tiles[tile_ix].backdrop, backdrop);
                    }
                    var next_xray = last_xray;
                    if y + 1 < y1 {
                        let tile_y1 = f32(y + 1) * f32(TILE_HEIGHT);
                        let x_edge = lp0.x + (tile_y1 - lp0.y) * invslope;
                        next_xray = i32(floor(x_edge * SX));
                    }
                    let min_xray = min(xray, next_xray);
                    let max_xray = max(xray, next_xray);
                    var xx0 = min(i32(floor(xc - c)), min_xray);
                    var xx1 = max(i32(ceil(xc + c)), max_xray + 1);
                    xx0 = clamp(xx0, x0, x1);
                    xx1 = clamp(xx1, x0, x1);
                    var tile_seg: Segment;
                    for (var x = xx0; x < xx1; x += 1) {
                        let tile_x0 = f32(x) * f32(TILE_WIDTH);
                        let tile_ix = base + x;
                        // allocate segment, insert linked list
                        let seg_ix = alloc_segment();
                        let old = atomicExchange(&tiles[tile_ix].segments, seg_ix);
                        tile_seg.origin = lp0;
                        tile_seg.delta = dp;
                        var y_edge = 0.0;
                        if !is_stroke {
                            y_edge = mix(lp0.y, lp1.y, (tile_x0 - lp0.x) * recip_dx);
                            if xymin.x < tile_x0 {
                                let p = vec2(tile_x0, y_edge);
                                if dp.x < 0.0 {
                                    tile_seg.delta = p - lp0;
                                } else {
                                    tile_seg.origin = p;
                                    tile_seg.delta = lp1 - p;
                                }
                                if tile_seg.delta.x == 0.0 {
                                    tile_seg.delta.x = sign(dp.x) * 1e-9;
                                }
                            }
                            if x <= min_xray || max_xray < x {
                                y_edge = 1e9;
                            }
                        }
                        tile_seg.y_edge = y_edge;
                        tile_seg.next = old;
                        segments[seg_ix] = tile_seg;
                    }
                    xc += b;
                    base += stride;
                    xray = next_xray;
                }
                n_out += 1u;
                val_target += v_step;
                lp0 = lp1;
            }
            val_sum += params.val;
            qp0 = qp2;
        }
    }
}
