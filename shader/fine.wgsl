// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Fine rasterizer. This can run in simple (just path rendering) and full
// modes, controllable by #define.
//
// To enable multisampled rendering, turn on both the msaa ifdef and one of msaa8
// or msaa16.

struct Tile {
    backdrop: i32,
    segments: u32,
}

#import segment
#import config

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> segments: array<Segment>;

#import blend
#import ptcl

let GRADIENT_WIDTH = 512;

@group(0) @binding(2)
var<storage> ptcl: array<u32>;

@group(0) @binding(3)
var<storage> info: array<u32>;

@group(0) @binding(4)
var output: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(5)
var gradients: texture_2d<f32>;

@group(0) @binding(6)
var image_atlas: texture_2d<f32>;

#ifdef msaa8
let MASK_WIDTH = 32u;
let MASK_HEIGHT = 32u;
let SH_SAMPLES_SIZE = 256u;
let SAMPLE_WORDS_PER_PIXEL = 1u;
// This might be better in uniform, but that has 16 byte alignment
@group(0) @binding(7)
var<storage> mask_lut: array<u32, 256u>;
#endif

#ifdef msaa16
let MASK_WIDTH = 64u;
let MASK_HEIGHT = 64u;
let SH_SAMPLES_SIZE = 512u;
let SAMPLE_WORDS_PER_PIXEL = 2u;
@group(0) @binding(7)
var<storage> mask_lut: array<u32, 2048u>;
#endif

#ifdef msaa
let WG_SIZE = 64u;
var<workgroup> sh_count: array<u32, WG_SIZE>;

// This is 8 winding numbers packed to a u32, 4 bits per sample
var<workgroup> sh_winding: array<atomic<u32>, 32u>;
// Same packing, one group of 8 per pixel
var<workgroup> sh_samples: array<atomic<u32>, SH_SAMPLES_SIZE>;
// Same packing, accumulating winding numbers for vertical edge crossings
var<workgroup> sh_winding_y: array<atomic<u32>, 2u>;

// number of integer cells spanned by interval defined by a, b
fn span(a: f32, b: f32) -> u32 {
    return u32(max(ceil(max(a, b)) - floor(min(a, b)), 1.0));
}

let SEG_SIZE = 5u;

// See cpu_shaders/util.rs for explanation of these.
let ONE_MINUS_ULP: f32 = 0.99999994;
let ROBUST_EPSILON: f32 = 2e-7;

// New multisampled algorithm.
//
// FIXME: This should return an array when https://github.com/gfx-rs/naga/issues/1930 is fixed.
fn fill_path_ms(fill: CmdFill, wg_id: vec2<u32>, local_id: vec2<u32>, result: ptr<function, array<f32, PIXELS_PER_THREAD>>) {
    let n_segs = fill.size_and_rule >> 1u;
    let even_odd = (fill.size_and_rule & 1u) != 0u;
    let tile_origin = vec2(f32(wg_id.x) * f32(TILE_HEIGHT), f32(wg_id.y) * f32(TILE_WIDTH));
    let th_ix = local_id.y * (TILE_WIDTH / PIXELS_PER_THREAD) + local_id.x;
    if th_ix < 32u {
        if th_ix < 2u {
            atomicStore(&sh_winding_y[th_ix], 0x88888888u);
        }
        atomicStore(&sh_winding[th_ix], 0x88888888u);
    }
    let sample_count = PIXELS_PER_THREAD * SAMPLE_WORDS_PER_PIXEL;
    for (var i = 0u; i < sample_count; i++) {
        atomicStore(&sh_samples[th_ix * sample_count + i], 0x88888888u);
    }
    workgroupBarrier();
    let n_batch = (n_segs + (WG_SIZE - 1u)) / WG_SIZE;
    for (var batch = 0u; batch < n_batch; batch++) {
        let seg_ix = batch * WG_SIZE + th_ix;
        let seg_off = fill.seg_data + seg_ix;
        var count = 0u;
        let slice_size = min(n_segs - batch * WG_SIZE, WG_SIZE);
        // TODO: might save a register rewriting this in terms of limit
        if th_ix < slice_size {
            let segment = segments[seg_off];
            // Note: coords relative to tile origin probably a good idea in coarse path,
            // especially as f16 would work. But keeping existing scheme for compatibility.
            let xy0 = segment.origin - tile_origin;
            let xy1 = xy0 + segment.delta;
            var y_edge_f = f32(TILE_HEIGHT);
            var delta = select(-1, 1, xy1.x <= xy0.x);
            if xy0.x == 0.0 && xy1.x == 0.0 {
                if xy0.y == 0.0 {
                    y_edge_f = 0.0;
                } else if xy1.y == 0.0 {
                    y_edge_f = 0.0;
                    delta = -delta;
                }
            } else {
                if xy0.x == 0.0 {
                    if xy0.y != 0.0 {
                        y_edge_f = xy0.y;
                    }
                } else if xy1.x == 0.0 && xy1.y != 0.0 {
                    y_edge_f = xy1.y;
                }
                // discard horizontal lines aligned to pixel grid
                if !(xy0.y == xy1.y && xy0.y == floor(xy0.y)) {
                    count = span(xy0.x, xy1.x) + span(xy0.y, xy1.y) - 1u;
                }
            }
            let y_edge = u32(ceil(y_edge_f));
            if y_edge < TILE_HEIGHT {
                atomicAdd(&sh_winding_y[y_edge >> 3u], u32(delta) << ((y_edge & 7u) << 2u));
            }
        }
        // workgroup prefix sum of counts
        sh_count[th_ix] = count;
        let lg_n = firstLeadingBit(slice_size * 2u - 1u);
        for (var i = 0u; i < lg_n; i++) {
            workgroupBarrier();
            if th_ix >= 1u << i {
                count += sh_count[th_ix - (1u << i)];
            }
            workgroupBarrier();
            sh_count[th_ix] = count;
        }
        let total = workgroupUniformLoad(&sh_count[slice_size - 1u]);
        for (var i = th_ix; i < total; i += WG_SIZE) {
            // binary search to find pixel
            var lo = 0u;
            var hi = slice_size;
            let goal = i;
            while hi > lo + 1u {
                let mid = (lo + hi) >> 1u;
                if goal >= sh_count[mid - 1u] {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let el_ix = lo;
            let last_pixel = i + 1u == sh_count[el_ix];
            let sub_ix = i - select(0u, sh_count[el_ix - 1u], el_ix > 0u);
            let seg_off = fill.seg_data + batch * WG_SIZE + el_ix;
            let segment = segments[seg_off];
            let xy0_in = segment.origin - tile_origin;
            let xy1_in = xy0_in + segment.delta;
            let is_down = xy1_in.y >= xy0_in.y;
            let xy0 = select(xy1_in, xy0_in, is_down);
            let xy1 = select(xy0_in, xy1_in, is_down);

            // Set up data for line rasterization
            // Note: this is duplicated work if total count exceeds a workgroup.
            // One alternative is to compute it in a separate dispatch.
            let dx = abs(xy1.x - xy0.x);
            let dy = xy1.y - xy0.y;
            let idxdy = 1.0 / (dx + dy);
            var a = dx * idxdy;
            let is_positive_slope = xy1.x >= xy0.x;
            let x_sign = select(-1.0, 1.0, is_positive_slope);
            let xt0 = floor(xy0.x * x_sign);
            let c = xy0.x * x_sign - xt0;
            let y0i = floor(xy0.y);
            let ytop = y0i + 1.0;
            let b = min((dy * c + dx * (ytop - xy0.y)) * idxdy, ONE_MINUS_ULP);
            let count_x = span(xy0.x, xy1.x) - 1u;
            let count = count_x + span(xy0.y, xy1.y);
            let robust_err = floor(a * (f32(count) - 1.0) + b) - f32(count_x);
            if robust_err != 0.0 {
                a -= ROBUST_EPSILON * sign(robust_err);
            }
            let x0i = i32(xt0 * x_sign + 0.5 * (x_sign - 1.0));
            // Use line equation to plot pixel coordinates

            let zf = a * f32(sub_ix) + b;
            let z = floor(zf);
            let x = x0i + i32(x_sign * z);
            let y = i32(y0i) + i32(sub_ix) - i32(z);
            var is_delta: bool;
            // We need to adjust winding number if slope is positive and there
            // is a crossing at the left edge of the pixel.
            var is_bump = false;
            let zp = floor(a * f32(sub_ix - 1u) + b);
            if sub_ix == 0u {
                is_delta = y0i == xy0.y && y0i != xy1.y;
                is_bump = xy0.x == 0.0;
            } else {
                is_delta = z == zp;
                is_bump = is_positive_slope && !is_delta;
            }
            let pix_ix = u32(y) * TILE_WIDTH + u32(x);
            if u32(x) < TILE_WIDTH - 1u && u32(y) < TILE_HEIGHT {
                let delta_pix = pix_ix + 1u;
                if is_delta {
                    let delta = select(u32(-1), 1u, is_down) << ((delta_pix & 7u) << 2u);
                    atomicAdd(&sh_winding[delta_pix >> 3u], delta);
                }
            }
            // Apply sample mask
            let mask_block = u32(is_positive_slope) * (MASK_WIDTH * MASK_HEIGHT / 2u);
            let half_height = f32(MASK_HEIGHT / 2u);
            let mask_row = floor(min(a * half_height, half_height - 1.0)) * f32(MASK_WIDTH);
            let mask_col = floor((zf - z) * f32(MASK_WIDTH));
            let mask_ix = mask_block + u32(mask_row + mask_col);
#ifdef msaa8
            var mask = mask_lut[mask_ix / 4u] >> ((mask_ix % 4u) * 8u);
            mask &= 0xffu;
            // Intersect with y half-plane masks
            if sub_ix == 0u && !is_bump {
                let mask_shift = u32(round(8.0 * (xy0.y - f32(y))));
                mask &= 0xffu << mask_shift;
            }
            if last_pixel && xy1.x != 0.0 {
                let mask_shift = u32(round(8.0 * (xy1.y - f32(y))));
                mask &= ~(0xffu << mask_shift);
            }
            let mask_a = mask | (mask << 6u);
            let mask_b = mask_a | (mask_a << 12u);
            let mask_exp = (mask_b & 0x1010101u) | ((mask_b << 3u) & 0x10101010u);
            var mask_signed = select(mask_exp, u32(-i32(mask_exp)), is_down);
            if is_bump {
                mask_signed += select(u32(-0x11111111), 0x1111111u, is_down);
            }
            atomicAdd(&sh_samples[pix_ix], mask_signed);
#endif
#ifdef msaa16
            var mask = mask_lut[mask_ix / 2u] >> ((mask_ix % 2u) * 16u);
            mask &= 0xffffu;
            // Intersect with y half-plane masks
            if sub_ix == 0u && !is_bump {
                let mask_shift = u32(round(16.0 * (xy0.y - f32(y))));
                mask &= 0xffffu << mask_shift;
            }
            if last_pixel && xy1.x != 0.0 {
                let mask_shift = u32(round(16.0 * (xy1.y - f32(y))));
                mask &= ~(0xffffu << mask_shift);
            }
            let mask0 = mask & 0xffu;
            let mask0_a = mask0 | (mask0 << 6u);
            let mask0_b = mask0_a | (mask0_a << 12u);
            let mask0_exp = (mask0_b & 0x1010101u) | ((mask0_b << 3u) & 0x10101010u);
            var mask0_signed = select(mask0_exp, u32(-i32(mask0_exp)), is_down);
            let mask1 = (mask >> 8u) & 0xffu;
            let mask1_a = mask1 | (mask1 << 6u);
            let mask1_b = mask1_a | (mask1_a << 12u);
            let mask1_exp = (mask1_b & 0x1010101u) | ((mask1_b << 3u) & 0x10101010u);
            var mask1_signed = select(mask1_exp, u32(-i32(mask1_exp)), is_down);
            if is_bump {
                let bump_delta = select(u32(-0x11111111), 0x1111111u, is_down);
                mask0_signed += bump_delta;
                mask1_signed += bump_delta;
            }
            atomicAdd(&sh_samples[pix_ix * 2u], mask0_signed);
            atomicAdd(&sh_samples[pix_ix * 2u + 1u], mask1_signed);
#endif
        }
        workgroupBarrier();
    }
    var area: array<f32, PIXELS_PER_THREAD>;
    let major = (th_ix * PIXELS_PER_THREAD) >> 3u;
    var packed_w = atomicLoad(&sh_winding[major]);
    // Prefix sum of packed 4 bit values within u32
    packed_w += (packed_w - 0x8888888u) << 4u;
    packed_w += (packed_w - 0x888888u) << 8u;
    packed_w += (packed_w - 0x8888u) << 16u;
    // Note: could probably do bias in one go, but it would be inscrutable
    if (major & 1u) != 0u {
        // We could use shmem to communicate the value from another thread;
        // if we had subgroups that would almost certainly be the most
        // efficient way. But we just calculate again for simplicity.
        var last_packed = atomicLoad(&sh_winding[major - 1u]);
        last_packed += (last_packed - 0x8888888u) << 4u;
        last_packed += (last_packed - 0x888888u) << 8u;
        last_packed += (last_packed - 0x8888u) << 16u;
        let bump = ((last_packed >> 28u) - 8u) * 0x11111111u;
        packed_w += bump;
    }
    var packed_y = atomicLoad(&sh_winding_y[local_id.y >> 3u]);
    packed_y += (packed_y - 0x8888888u) << 4u;
    packed_y += (packed_y - 0x888888u) << 8u;
    packed_y += (packed_y - 0x8888u) << 16u;
    if th_ix == 0u {
        atomicStore(&sh_winding_y[0], packed_y);
    }
    workgroupBarrier();
    var wind_y = (packed_y >> ((local_id.y & 7u) << 2u)) - 8u;
    if local_id.y >= 8u {
        wind_y += (atomicLoad(&sh_winding_y[0]) >> 28u) - 8u;
    }

    for (var i = 0u; i < PIXELS_PER_THREAD; i++) {
        let pix_ix = th_ix * PIXELS_PER_THREAD + i;
        let minor = pix_ix & 7u;
        //let nonzero = ((packed_w >> (minor << 2u)) & 0xfu) != u32(8 + backdrop);
        // TODO: math might be off here
        let expected_zero = (((packed_w >> (minor * 4u)) + wind_y) & 0xfu) - u32(fill.backdrop);
        if expected_zero >= 16u {
            area[i] = 1.0;
        } else {
#ifdef msaa8
            let samples = atomicLoad(&sh_samples[pix_ix]);
            let xored = (expected_zero * 0x11111111u) ^ samples;
            // Each 4-bit nibble in xored is 0 for winding = 0, nonzero otherwise
            let xored2 = xored | (xored * 2u);
            let xored4 = xored2 | (xored2 * 4u);
            area[i] = f32(countOneBits(xored4 & 0x88888888u)) * 0.125;
#endif
#ifdef msaa16
            let samples0 = atomicLoad(&sh_samples[pix_ix * 2u]);
            let samples1 = atomicLoad(&sh_samples[pix_ix * 2u + 1u]);
            let xored0 = (expected_zero * 0x11111111u) ^ samples0;
            let xored0_2 = xored0 | (xored0 * 2u);
            let xored1 = (expected_zero * 0x11111111u) ^ samples1;
            let xored1_2 = xored1 | (xored1 >> 1u);
            let xored2 = (xored0_2 & 0xAAAAAAAAu) | (xored1_2 & 0x55555555u);
            let xored4 = xored2 | (xored2 * 4u);
            area[i] = f32(countOneBits(xored4 & 0xCCCCCCCCu)) * 0.0625;
#endif
        }
    }
    *result = area;
}
#endif

fn read_fill(cmd_ix: u32) -> CmdFill {
    let size_and_rule = ptcl[cmd_ix + 1u];
    let seg_data = ptcl[cmd_ix + 2u];
    let backdrop = i32(ptcl[cmd_ix + 3u]);
    return CmdFill(size_and_rule, seg_data, backdrop);
}

fn read_color(cmd_ix: u32) -> CmdColor {
    let rgba_color = ptcl[cmd_ix + 1u];
    return CmdColor(rgba_color);
}

fn read_lin_grad(cmd_ix: u32) -> CmdLinGrad {
    let index_mode = ptcl[cmd_ix + 1u];
    let index = index_mode >> 2u;
    let extend_mode = index_mode & 0x3u;
    let info_offset = ptcl[cmd_ix + 2u];
    let line_x = bitcast<f32>(info[info_offset]);
    let line_y = bitcast<f32>(info[info_offset + 1u]);
    let line_c = bitcast<f32>(info[info_offset + 2u]);
    return CmdLinGrad(index, extend_mode, line_x, line_y, line_c);
}

fn read_rad_grad(cmd_ix: u32) -> CmdRadGrad {
    let index_mode = ptcl[cmd_ix + 1u];
    let index = index_mode >> 2u;
    let extend_mode = index_mode & 0x3u;
    let info_offset = ptcl[cmd_ix + 2u];
    let m0 = bitcast<f32>(info[info_offset]);
    let m1 = bitcast<f32>(info[info_offset + 1u]);
    let m2 = bitcast<f32>(info[info_offset + 2u]);
    let m3 = bitcast<f32>(info[info_offset + 3u]);
    let matrx = vec4(m0, m1, m2, m3);
    let xlat = vec2(bitcast<f32>(info[info_offset + 4u]), bitcast<f32>(info[info_offset + 5u]));
    let focal_x = bitcast<f32>(info[info_offset + 6u]);
    let radius = bitcast<f32>(info[info_offset + 7u]);
    let flags_kind = info[info_offset + 8u];
    let flags = flags_kind >> 3u;
    let kind = flags_kind & 0x7u;
    return CmdRadGrad(index, extend_mode, matrx, xlat, focal_x, radius, kind, flags);
}

fn read_image(cmd_ix: u32) -> CmdImage {
    let info_offset = ptcl[cmd_ix + 1u];
    let m0 = bitcast<f32>(info[info_offset]);
    let m1 = bitcast<f32>(info[info_offset + 1u]);
    let m2 = bitcast<f32>(info[info_offset + 2u]);
    let m3 = bitcast<f32>(info[info_offset + 3u]);
    let matrx = vec4(m0, m1, m2, m3);
    let xlat = vec2(bitcast<f32>(info[info_offset + 4u]), bitcast<f32>(info[info_offset + 5u]));
    let xy = info[info_offset + 6u];
    let width_height = info[info_offset + 7u];
    // The following are not intended to be bitcasts
    let x = f32(xy >> 16u);
    let y = f32(xy & 0xffffu);
    let width = f32(width_height >> 16u);
    let height = f32(width_height & 0xffffu);
    return CmdImage(matrx, xlat, vec2(x, y), vec2(width, height));
}

fn read_end_clip(cmd_ix: u32) -> CmdEndClip {
    let blend = ptcl[cmd_ix + 1u];
    let alpha = bitcast<f32>(ptcl[cmd_ix + 2u]);
    return CmdEndClip(blend, alpha);
}

fn extend_mode(t: f32, mode: u32) -> f32 {
    let EXTEND_PAD = 0u;
    let EXTEND_REPEAT = 1u;
    let EXTEND_REFLECT = 2u;
    switch mode {
        // EXTEND_PAD
        case 0u: {
            return clamp(t, 0.0, 1.0);
        }
        // EXTEND_REPEAT
        case 1u: {
            return fract(t);
        }
        // EXTEND_REFLECT
        default: {
            return abs(t - 2.0 * round(0.5 * t));
        }
    }
}

let PIXELS_PER_THREAD = 4u;

// Analytic area antialiasing.
//
// This is currently dead code if msaa is enabled, but it would be fairly straightforward
// to wire this so it's a dynamic choice (even per-path).
//
// FIXME: This should return an array when https://github.com/gfx-rs/naga/issues/1930 is fixed.
fn fill_path(fill: CmdFill, xy: vec2<f32>, result: ptr<function, array<f32, PIXELS_PER_THREAD>>) {
    let n_segs = fill.size_and_rule >> 1u;
    let even_odd = (fill.size_and_rule & 1u) != 0u;
    var area: array<f32, PIXELS_PER_THREAD>;
    let backdrop_f = f32(fill.backdrop);
    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
        area[i] = backdrop_f;
    }
    for (var i = 0u; i < n_segs; i++) {
        let seg_off = fill.seg_data + i;
        let segment = segments[seg_off];
        let y = segment.origin.y - xy.y;
        let y0 = clamp(y, 0.0, 1.0);
        let y1 = clamp(y + segment.delta.y, 0.0, 1.0);
        let dy = y0 - y1;
        if dy != 0.0 {
            let vec_y_recip = 1.0 / segment.delta.y;
            let t0 = (y0 - y) * vec_y_recip;
            let t1 = (y1 - y) * vec_y_recip;
            let startx = segment.origin.x - xy.x;
            let x0 = startx + t0 * segment.delta.x;
            let x1 = startx + t1 * segment.delta.x;
            let xmin0 = min(x0, x1);
            let xmax0 = max(x0, x1);
            for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                let i_f = f32(i);
                let xmin = min(xmin0 - i_f, 1.0) - 1.0e-6;
                let xmax = xmax0 - i_f;
                let b = min(xmax, 1.0);
                let c = max(b, 0.0);
                let d = max(xmin, 0.0);
                let a = (b + 0.5 * (d * d - c * c) - xmin) / (xmax - xmin);
                area[i] += a * dy;
            }
        }
        let y_edge = sign(segment.delta.x) * clamp(xy.y - segment.y_edge + 1.0, 0.0, 1.0);
        for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
            area[i] += y_edge;
        }
    }
    if even_odd {
        // even-odd winding rule
        for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
            let a = area[i];
            area[i] = abs(a - 2.0 * round(0.5 * a));
        }
    } else {
        // non-zero winding rule
        for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
            area[i] = min(abs(area[i]), 1.0);
        }
    }
    *result = area;
}

// The X size should be 16 / PIXELS_PER_THREAD
@compute @workgroup_size(4, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let tile_ix = wg_id.y * config.width_in_tiles + wg_id.x;
    let xy = vec2(f32(global_id.x * PIXELS_PER_THREAD), f32(global_id.y));
#ifdef full
    var rgba: array<vec4<f32>, PIXELS_PER_THREAD>;
    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
        rgba[i] = unpack4x8unorm(config.base_color).wzyx;
    }
    var blend_stack: array<array<u32, PIXELS_PER_THREAD>, BLEND_STACK_SPLIT>;
    var clip_depth = 0u;
    var area: array<f32, PIXELS_PER_THREAD>;
    var cmd_ix = tile_ix * PTCL_INITIAL_ALLOC;
    let blend_offset = ptcl[cmd_ix];
    cmd_ix += 1u;
    // main interpretation loop
    while true {
        let tag = ptcl[cmd_ix];
        if tag == CMD_END {
            break;
        }
        switch tag {
            // CMD_FILL
            case 1u: {
                let fill = read_fill(cmd_ix);
#ifdef msaa
                fill_path_ms(fill, wg_id.xy, local_id.xy, &area);
#else
                fill_path(fill, xy, &area);
#endif
                cmd_ix += 4u;
            }
            // CMD_STROKE
            case 2u: {
                // Stroking in fine rasterization is disabled, as strokes will be expanded
                // to fills earlier in the pipeline. This implementation is a stub, just to
                // keep the shader from crashing.
                for (var i = 0u; i < PIXELS_PER_THREAD; i++) {
                    area[i] = 0.0;
                }
                cmd_ix += 3u;
            }
            // CMD_SOLID
            case 3u: {
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    area[i] = 1.0;
                }
                cmd_ix += 1u;
            }
            // CMD_COLOR
            case 5u: {
                let color = read_color(cmd_ix);
                let fg = unpack4x8unorm(color.rgba_color).wzyx;
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    let fg_i = fg * area[i];
                    rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                }
                cmd_ix += 2u;
            }
            // CMD_LIN_GRAD
            case 6u: {
                let lin = read_lin_grad(cmd_ix);
                let d = lin.line_x * xy.x + lin.line_y * xy.y + lin.line_c;
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    let my_d = d + lin.line_x * f32(i);
                    let x = i32(round(extend_mode(my_d, lin.extend_mode) * f32(GRADIENT_WIDTH - 1)));
                    let fg_rgba = textureLoad(gradients, vec2(x, i32(lin.index)), 0);
                    let fg_i = fg_rgba * area[i];
                    rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                }
                cmd_ix += 3u;
            }
            // CMD_RAD_GRAD
            case 7u: {
                let rad = read_rad_grad(cmd_ix);
                let focal_x = rad.focal_x;
                let radius = rad.radius;
                let is_strip = rad.kind == RAD_GRAD_KIND_STRIP;
                let is_circular = rad.kind == RAD_GRAD_KIND_CIRCULAR;
                let is_focal_on_circle = rad.kind == RAD_GRAD_KIND_FOCAL_ON_CIRCLE;
                let is_swapped = (rad.flags & RAD_GRAD_SWAPPED) != 0u;
                let r1_recip = select(1.0 / radius, 0.0, is_circular);
                let less_scale = select(1.0, -1.0, is_swapped || (1.0 - focal_x) < 0.0);
                let t_sign = sign(1.0 - focal_x);
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    let my_xy = vec2(xy.x + f32(i), xy.y);
                    let local_xy = rad.matrx.xy * my_xy.x + rad.matrx.zw * my_xy.y + rad.xlat;
                    let x = local_xy.x;
                    let y = local_xy.y;
                    let xx = x * x;
                    let yy = y * y;
                    var t = 0.0;
                    var is_valid = true;
                    if is_strip {
                        let a = radius - yy;
                        t = sqrt(a) + x;
                        is_valid = a >= 0.0;
                    } else if is_focal_on_circle {
                        t = (xx + yy) / x;
                        is_valid = t >= 0.0 && x != 0.0;
                    } else if radius > 1.0 {
                        t = sqrt(xx + yy) - x * r1_recip;
                    } else { // radius < 1.0
                        let a = xx - yy;
                        t = less_scale * sqrt(a) - x * r1_recip;
                        is_valid = a >= 0.0 && t >= 0.0;
                    }
                    if is_valid {
                        t = extend_mode(focal_x + t_sign * t, rad.extend_mode);
                        t = select(t, 1.0 - t, is_swapped);
                        let x = i32(round(t * f32(GRADIENT_WIDTH - 1)));
                        let fg_rgba = textureLoad(gradients, vec2(x, i32(rad.index)), 0);
                        let fg_i = fg_rgba * area[i];
                        rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                    }
                }
                cmd_ix += 3u;
            }
            // CMD_IMAGE
            case 8u: {
                let image = read_image(cmd_ix);
                let atlas_extents = image.atlas_offset + image.extents;
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    let my_xy = vec2(xy.x + f32(i), xy.y);
                    let atlas_uv = image.matrx.xy * my_xy.x + image.matrx.zw * my_xy.y + image.xlat + image.atlas_offset;
                    // This currently clips to the image bounds. TODO: extend modes
                    if all(atlas_uv < atlas_extents) && area[i] != 0.0 {
                        let uv_quad = vec4(max(floor(atlas_uv), image.atlas_offset), min(ceil(atlas_uv), atlas_extents));
                        let uv_frac = fract(atlas_uv);
                        let a = premul_alpha(textureLoad(image_atlas, vec2<i32>(uv_quad.xy), 0));
                        let b = premul_alpha(textureLoad(image_atlas, vec2<i32>(uv_quad.xw), 0));
                        let c = premul_alpha(textureLoad(image_atlas, vec2<i32>(uv_quad.zy), 0));
                        let d = premul_alpha(textureLoad(image_atlas, vec2<i32>(uv_quad.zw), 0));
                        let fg_rgba = mix(mix(a, b, uv_frac.y), mix(c, d, uv_frac.y), uv_frac.x);
                        let fg_i = fg_rgba * area[i];
                        rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                    }
                }
                cmd_ix += 2u;
            }
            // CMD_BEGIN_CLIP
            case 9u: {
                if clip_depth < BLEND_STACK_SPLIT {
                    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                        blend_stack[clip_depth][i] = pack4x8unorm(rgba[i]);
                        rgba[i] = vec4(0.0);
                    }
                } else {
                    // TODO: spill to memory
                }
                clip_depth += 1u;
                cmd_ix += 1u;
            }
            // CMD_END_CLIP
            case 10u: {
                let end_clip = read_end_clip(cmd_ix);
                clip_depth -= 1u;
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    var bg_rgba: u32;
                    if clip_depth < BLEND_STACK_SPLIT {
                        bg_rgba = blend_stack[clip_depth][i];
                    } else {
                        // load from memory
                    }
                    let bg = unpack4x8unorm(bg_rgba);
                    let fg = rgba[i] * area[i] * end_clip.alpha;
                    rgba[i] = blend_mix_compose(bg, fg, end_clip.blend);
                }
                cmd_ix += 3u;
            }
            // CMD_JUMP
            case 11u: {
                cmd_ix = ptcl[cmd_ix + 1u];
            }
            default: {}
        }
    }
    let xy_uint = vec2<u32>(xy);
    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
        let coords = xy_uint + vec2(i, 0u);
        if coords.x < config.target_width && coords.y < config.target_height {
            let fg = rgba[i];
            // Max with a small epsilon to avoid NaNs
            let a_inv = 1.0 / max(fg.a, 1e-6);
            let rgba_sep = vec4(fg.rgb * a_inv, fg.a);
            textureStore(output, vec2<i32>(coords), rgba_sep);
        }
    } 
#else
    let tile = tiles[tile_ix];
    var area: array<f32, PIXELS_PER_THREAD>;
    fill_path(tile, xy, &area);

    let xy_uint = vec2<u32>(xy);
    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
        let coords = xy_uint + vec2(i, 0u);
        if coords.x < config.target_width && coords.y < config.target_height {
            textureStore(output, vec2<i32>(coords), vec4(area[i]));
        }
    }
#endif
}

fn premul_alpha(rgba: vec4<f32>) -> vec4<f32> {
    return vec4(rgba.rgb * rgba.a, rgba.a);
}
