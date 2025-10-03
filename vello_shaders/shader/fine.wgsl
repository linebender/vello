// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Fine rasterizer.
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

const GRADIENT_WIDTH = 512;

const IMAGE_QUALITY_LOW = 0u;
const IMAGE_QUALITY_MEDIUM = 1u;
const IMAGE_QUALITY_HIGH = 2u;

const LUMINANCE_MASK_LAYER = 0x10000u;

@group(0) @binding(2)
var<storage> ptcl: array<u32>;

@group(0) @binding(3)
var<storage> info: array<u32>;

@group(0) @binding(4)
var<storage, read_write> blend_spill: array<u32>;

@group(0) @binding(5)
var output: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(6)
var gradients: texture_2d<f32>;

@group(0) @binding(7)
var image_atlas: texture_2d<f32>;

// MSAA-only bindings and utilities
#ifdef msaa

const MASK_LUT_INDEX: u32 = 8;

#ifdef msaa8
const MASK_WIDTH = 32u;
const MASK_HEIGHT = 32u;
const SH_SAMPLES_SIZE = 512u;
const SAMPLE_WORDS_PER_PIXEL = 2u;
// This might be better in uniform, but that has 16 byte alignment
@group(0) @binding(MASK_LUT_INDEX)
var<storage> mask_lut: array<u32, 256u>;
#endif

#ifdef msaa16
const MASK_WIDTH = 64u;
const MASK_HEIGHT = 64u;
const SH_SAMPLES_SIZE = 1024u;
const SAMPLE_WORDS_PER_PIXEL = 4u;
@group(0) @binding(MASK_LUT_INDEX)
var<storage> mask_lut: array<u32, 2048u>;
#endif

const WG_SIZE = 64u;
var<workgroup> sh_count: array<u32, WG_SIZE>;

// This array contains the winding number of the top left corner of each
// 16 pixel wide row of pixels, relative to the top left corner of the row
// immediately above.
//
// The values are stored packed, as 4 8-bit subwords in a 32 bit word.
// The values are biased signed integers, with 0x80 representing a winding
// number of 0, so that the range of -128 to 127 (inclusive) can be stored
// without carry.
//
// For the even-odd case, the same storage is repurposed, so that a single
// word contains 16 one-bit winding parity values packed to the word.
var<workgroup> sh_winding_y: array<atomic<u32>, 4u>;
// This array contains the winding number of the top left corner of each
// 16 pixel wide row of pixels, relative to the top left corner of the tile.
// It is expanded from sh_winding_y by inclusive prefix sum.
var<workgroup> sh_winding_y_prefix: array<atomic<u32>, 4u>;
// This array contains winding numbers of the top left corner of each
// pixel, relative to the top left corner of the enclosing 16 pixel
// wide row.
//
// During winding number accumulation, it stores a delta (winding number
// relative to the pixel immediately to the left), then expanded using
// prefix sum and reusing the same storage.
//
// The encoding and packing is the same as `sh_winding_y`. For the even-odd
// case, only the first 16 values are used, and each word stores packed
// parity values for one row of pixels.
var<workgroup> sh_winding: array<atomic<u32>, 64u>;
// This array contains winding numbers of multiple sample points within
// a pixel, relative to the winding number of the top left corner of the
// pixel. The encoding and packing is the same as `sh_winding_y`.
var<workgroup> sh_samples: array<atomic<u32>, SH_SAMPLES_SIZE>;

// number of integer cells spanned by interval defined by a, b
fn span(a: f32, b: f32) -> u32 {
    return u32(max(ceil(max(a, b)) - floor(min(a, b)), 1.0));
}

const SEG_SIZE = 5u;

// See cpu_shaders/util.rs for explanation of these.
const ONE_MINUS_ULP: f32 = 0.99999994;
const ROBUST_EPSILON: f32 = 2e-7;

// Multisampled path rendering algorithm.
//
// FIXME: This could return an array when https://github.com/gfx-rs/naga/issues/1930 is fixed.
//
// Generally, this algorithm works in an accumulation phase followed by a
// resolving phase, with arrays in workgroup shared memory accumulating
// winding number deltas as the results of edge crossings detected in the
// path segments. Accumulation is in two stages: first a counting stage
// which computes the number of pixels touched by each line segment (with
// each thread processing one line segment), then a stage in which the
// deltas are bumped. Separating these two is a partition-wide prefix sum
// and a binary search to assign the work to threads in a load-balanced
// manner.
//
// The resolving phase is also two stages: prefix sums in both x and y
// directions, then counting nonzero winding numbers for all samples within
// all pixels in the tile.
//
// A great deal of SIMD within a register (SWAR) logic is used, as there
// are a great many winding numbers to be computed. The interested reader
// is invited to study the even-odd case first, as there only one bit is
// needed to represent a winding number parity, thus there is a lot less
// bit shifting, and less shuffling altogether.
fn fill_path_ms(fill: CmdFill, local_id: vec2<u32>, result: ptr<function, array<f32, PIXELS_PER_THREAD>>) {
    let even_odd = (fill.size_and_rule & 1u) != 0u;
    // This isn't a divergent branch because the fill parameters are workgroup uniform,
    // provably so because the ptcl buffer is bound read-only.
    if even_odd {
        fill_path_ms_evenodd(fill, local_id, result);
        return;
    }
    let n_segs = fill.size_and_rule >> 1u;
    let th_ix = local_id.y * (TILE_WIDTH / PIXELS_PER_THREAD) + local_id.x;
    // Initialize winding number arrays to a winding number of 0, which is 0x80 in an
    // 8 bit biased signed integer encoding.
    if th_ix < 64u {
        if th_ix < 4u {
            atomicStore(&sh_winding_y[th_ix], 0x80808080u);
        }
        atomicStore(&sh_winding[th_ix], 0x80808080u);
    }
    let sample_count = PIXELS_PER_THREAD * SAMPLE_WORDS_PER_PIXEL;
    for (var i = 0u; i < sample_count; i++) {
        atomicStore(&sh_samples[th_ix * sample_count + i], 0x80808080u);
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
            let xy0 = segment.point0;
            let xy1 = segment.point1;
            var y_edge_f = f32(TILE_HEIGHT);
            var delta = select(-1, 1, xy1.x <= xy0.x);
            if xy0.x == 0.0 {
                y_edge_f = xy0.y;
            } else if xy1.x == 0.0 {
                y_edge_f = xy1.y;
            }
            // discard horizontal lines aligned to pixel grid
            if !(xy0.y == xy1.y && xy0.y == floor(xy0.y)) {
                count = span(xy0.x, xy1.x) + span(xy0.y, xy1.y) - 1u;
            }
            let y_edge = u32(ceil(y_edge_f));
            if y_edge < TILE_HEIGHT {
                atomicAdd(&sh_winding_y[y_edge >> 2u], u32(delta) << ((y_edge & 3u) << 3u));
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
            // Coordinates are relative to tile origin
            let xy0_in = segment.point0;
            let xy1_in = segment.point1;
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
            // is_positive_slope is true for \ and | slopes, false for /. For
            // horizontal lines, it follows the original data.
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
            // is_delta captures whether the line crosses the top edge of this
            // pixel. If so, then a delta is added to `sh_winding`, followed by
            // a prefix sum, so that a winding number delta is applied to all
            // pixels to the right of this one.
            var is_delta: bool;
            // is_bump captures whether x0 crosses the left edge of this pixel.
            var is_bump = false;
            let zp = floor(a * f32(sub_ix - 1u) + b);
            if sub_ix == 0u {
                // The first (top-most) pixel in the line. It is considered to be
                // a line crossing when it touches the top of the pixel.
                //
                // Note: horizontal lines aligned to the pixel grid have already
                // been discarded.
                is_delta = y0i == xy0.y;
                // The pixel is counted as a left edge crossing only at the left
                // edge of the tile (and when it is not the top left corner,
                // using logic analogous to tiling).
                is_bump = xy0.x == 0.0 && y0i != xy0.y;
            } else {
                // Pixels other than the first are a crossing at the top or on
                // the side, based on the conservative line rasterization. When
                // positive slope, the crossing is on the left.
                is_delta = z == zp;
                is_bump = is_positive_slope && !is_delta;
            }
            let pix_ix = u32(y) * TILE_WIDTH + u32(x);
            if u32(x) < TILE_WIDTH - 1u && u32(y) < TILE_HEIGHT {
                let delta_pix = pix_ix + 1u;
                if is_delta {
                    let delta = select(u32(-1i), 1u, is_down) << ((delta_pix & 3u) << 3u);
                    atomicAdd(&sh_winding[delta_pix >> 2u], delta);
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
            // Expand an 8 bit mask value to 8 1-bit values, packed 4 to a subword,
            // so that two words are used to represent the result. An efficient
            // technique is carry-less multiplication by 0b10_0000_0100_0000_1000_0001
            // followed by and-masking to extract bit in position 4 * k.
            //
            // See https://en.wikipedia.org/wiki/Carry-less_product
            let mask_a = mask ^ (mask << 7u);
            let mask_b = mask_a ^ (mask_a << 14u);
            let mask0_exp = mask_b & 0x1010101u;
            var mask0_signed = select(mask0_exp, u32(-i32(mask0_exp)), is_down);
            let mask1_exp = (mask_b >> 4u) & 0x1010101u;
            var mask1_signed = select(mask1_exp, u32(-i32(mask1_exp)), is_down);
            if is_bump {
                let bump_delta = select(u32(-0x1010101i), 0x1010101u, is_down);
                mask0_signed += bump_delta;
                mask1_signed += bump_delta;
            }
            atomicAdd(&sh_samples[pix_ix * 2u], mask0_signed);
            atomicAdd(&sh_samples[pix_ix * 2u + 1u], mask1_signed);
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
            // Similar logic as above, only a 16 bit mask is divided into
            // two 8 bit halves first, then each is expanded as above.
            // Mask is 0bABCD_EFGH_IJKL_MNOP. Expand to 4 32 bit words
            // mask0_exp will be 0b0000_000M_0000_000N_0000_000O_0000_000P
            // mask3_exp will be 0b0000_000A_0000_000B_0000_000C_0000_000D
            let mask0 = mask & 0xffu;
            // mask0_a = 0b0IJK_LMNO_*JKL_MNOP
            let mask0_a = mask0 ^ (mask0 << 7u);
            // mask0_b = 0b000I_JKLM_NO*J_KLMN_O*K_LMNO_*JKL_MNOP
            //                ^    ^    ^    ^   ^    ^    ^    ^
            let mask0_b = mask0_a ^ (mask0_a << 14u);
            let mask0_exp = mask0_b & 0x1010101u;
            var mask0_signed = select(mask0_exp, u32(-i32(mask0_exp)), is_down);
            let mask1_exp = (mask0_b >> 4u) & 0x1010101u;
            var mask1_signed = select(mask1_exp, u32(-i32(mask1_exp)), is_down);
            let mask1 = (mask >> 8u) & 0xffu;
            let mask1_a = mask1 ^ (mask1 << 7u);
            // mask1_a = 0b0ABC_DEFG_*BCD_EFGH
            let mask1_b = mask1_a ^ (mask1_a << 14u);
            // mask1_b = 0b000A_BCDE_FG*B_CDEF_G*C_DEFG_*BCD_EFGH
            let mask2_exp = mask1_b & 0x1010101u;
            var mask2_signed = select(mask2_exp, u32(-i32(mask2_exp)), is_down);
            let mask3_exp = (mask1_b >> 4u) & 0x1010101u;
            var mask3_signed = select(mask3_exp, u32(-i32(mask3_exp)), is_down);
            if is_bump {
                let bump_delta = select(u32(-0x1010101i), 0x1010101u, is_down);
                mask0_signed += bump_delta;
                mask1_signed += bump_delta;
                mask2_signed += bump_delta;
                mask3_signed += bump_delta;
            }
            atomicAdd(&sh_samples[pix_ix * 4u], mask0_signed);
            atomicAdd(&sh_samples[pix_ix * 4u + 1u], mask1_signed);
            atomicAdd(&sh_samples[pix_ix * 4u + 2u], mask2_signed);
            atomicAdd(&sh_samples[pix_ix * 4u + 3u], mask3_signed);
#endif
        }
        workgroupBarrier();
    }
    var area: array<f32, PIXELS_PER_THREAD>;
    let major = (th_ix * PIXELS_PER_THREAD) >> 2u;
    var packed_w = atomicLoad(&sh_winding[major]);
    // Compute prefix sums of both `sh_winding` and `sh_winding_y`. Both
    // use the same technique. First, a per-word prefix sum is computed
    // of the 4 subwords within each word. The last subword is the sum
    // (reduction) of that group of 4 values, and is stored to shared
    // memory for broadcast to other threads. Then each thread computes
    // the prefix by adding the preceding reduced values.
    //
    // Addition of 2 biased signed values is accomplished by adding the
    // values, then subtracting the bias.
    packed_w += (packed_w - 0x808080u) << 8u;
    packed_w += (packed_w - 0x8080u) << 16u;
    var packed_y = atomicLoad(&sh_winding_y[local_id.y >> 2u]);
    packed_y += (packed_y - 0x808080u) << 8u;
    packed_y += (packed_y - 0x8080u) << 16u;
    var wind_y = (packed_y >> ((local_id.y & 3u) << 3u)) - 0x80u;
    if (local_id.y & 3u) == 3u && local_id.x == 0u {
        let prefix_y = wind_y;
        atomicStore(&sh_winding_y_prefix[local_id.y >> 2u], prefix_y);
    }
    let prefix_x = ((packed_w >> 24u) - 0x80u) * 0x1010101u;
    // reuse sh_winding to store prefix as well
    atomicStore(&sh_winding[major], prefix_x);
    workgroupBarrier();
    for (var i = (major & ~3u); i < major; i++) {
        packed_w += atomicLoad(&sh_winding[i]);
    }
    // packed_w now contains the winding numbers for a slice of 4 pixels,
    // each relative to the top left of the row.
    for (var i = 0u; i < (local_id.y >> 2u); i++) {
        wind_y += atomicLoad(&sh_winding_y_prefix[i]);
    }
    // wind_y now contains the winding number of the top left of the row of
    // pixels relative to the top left of the tile. Note that this is actually
    // a signed quantity stored without bias.

    // The winding number of a sample point is the sum of four levels of
    // hierarchy:
    // * The winding number of the top left of the tile (backdrop)
    // * The winding number of the pixel row relative to tile (wind_y)
    // * The winding number of the pixel relative to row (packed_w)
    // * The winding number of the sample relative to pixel (sh_samples)
    //
    // Conceptually, we want to compute each of these total winding numbers
    // for each sample within a pixel, then count the number that are non-zero.
    // However, we apply a shortcut, partly to make the computation more
    // efficient, and partly to avoid overflow of intermediate results.
    //
    // Here's the technique that's used. The `expected_zero` value contains
    // the *negation* of the sum of the first three levels of the hierarchy.
    // Thus, `sample - expected` is zero when the sum of all levels in the
    // hierarchy is zero, and this is true when `sample = expected`. We
    // compute this using SWAR techniques as follows: we compute the xor of
    // all bits of `expected` (repeated to all subwords) against the packed
    // samples, then the or-reduction of the bits within each subword. This
    // value is 1 when the values are unequal, thus the sum is nonzero, and
    // 0 when the sum is zero. These bits are then masked and counted.

    for (var i = 0u; i < PIXELS_PER_THREAD; i++) {
        let pix_ix = th_ix * PIXELS_PER_THREAD + i;
        let minor = i; // assumes PIXELS_PER_THREAD == 4
        let expected_zero = (((packed_w >> (minor * 8u)) + wind_y) & 0xffu) - u32(fill.backdrop);
        // When the expected_zero value exceeds the range of what can be stored
        // in a (biased) signed integer, then there is no sample value that can
        // be equal to the expected value, thus all resulting bits are 1.
        if expected_zero >= 256u {
            area[i] = 1.0;
        } else {
#ifdef msaa8
            let samples0 = atomicLoad(&sh_samples[pix_ix * 2u]);
            let samples1 = atomicLoad(&sh_samples[pix_ix * 2u + 1u]);
            let xored0 = (expected_zero * 0x1010101u) ^ samples0;
            let xored0_2 = xored0 | (xored0 * 2u);
            let xored1 = (expected_zero * 0x1010101u) ^ samples1;
            let xored1_2 = xored1 | (xored1 >> 1u);
            // xored2 contains 2-reductions from each word, interleaved
            let xored2 = (xored0_2 & 0xAAAAAAAAu) | (xored1_2 & 0x55555555u);
            // bits 4 * k + 2 and 4 * k + 3 contain 4-reductions
            let xored4 = xored2 | (xored2 * 4u);
            // bits 8 * k + 6 and 8 * k + 7 contain 8-reductions
            let xored8 = xored4 | (xored4 * 16u);
            area[i] = f32(countOneBits(xored8 & 0xC0C0C0C0u)) * 0.125;
#endif
#ifdef msaa16
            let samples0 = atomicLoad(&sh_samples[pix_ix * 4u]);
            let samples1 = atomicLoad(&sh_samples[pix_ix * 4u + 1u]);
            let samples2 = atomicLoad(&sh_samples[pix_ix * 4u + 2u]);
            let samples3 = atomicLoad(&sh_samples[pix_ix * 4u + 3u]);
            let xored0 = (expected_zero * 0x1010101u) ^ samples0;
            let xored0_2 = xored0 | (xored0 * 2u);
            let xored1 = (expected_zero * 0x1010101u) ^ samples1;
            let xored1_2 = xored1 | (xored1 >> 1u);
            // xored01 contains 2-reductions from words 0 and 1, interleaved
            let xored01 = (xored0_2 & 0xAAAAAAAAu) | (xored1_2 & 0x55555555u);
            // bits 4 * k + 2 and 4 * k + 3 contain 4-reductions
            let xored01_4 = xored01 | (xored01 * 4u);
            let xored2 = (expected_zero * 0x1010101u) ^ samples2;
            let xored2_2 = xored2 | (xored2 * 2u);
            let xored3 = (expected_zero * 0x1010101u) ^ samples3;
            let xored3_2 = xored3 | (xored3 >> 1u);
            // xored23 contains 2-reductions from words 2 and 3, interleaved
            let xored23 = (xored2_2 & 0xAAAAAAAAu) | (xored3_2 & 0x55555555u);
            // bits 4 * k and 4 * k + 1 contain 4-reductions
            let xored23_4 = xored23 | (xored23 >> 2u);
            // each bit is a 4-reduction, with values from all 4 words
            let xored4 = (xored01_4 & 0xCCCCCCCCu) | (xored23_4 & 0x33333333u);
            // bits 8 * k + {4, 5, 6, 7} contain 8-reductions
            let xored8 = xored4 | (xored4 * 16u);
            area[i] = f32(countOneBits(xored8 & 0xF0F0F0F0u)) * 0.0625;
#endif
        }
    }
    *result = area;
}

// Path rendering specialized to the even-odd rule.
//
// This proceeds very much the same as `fill_path_ms`, but is simpler because
// all winding numbers can be represented in one bit. Formally, addition is
// modulo 2, or, equivalently, winding numbers are elements of GF(2). One
// simplification is that we don't need to track the direction of crossings,
// as both have the same effect on winding number.
//
// TODO: factor some logic out to reduce code duplication.
fn fill_path_ms_evenodd(fill: CmdFill, local_id: vec2<u32>, result: ptr<function, array<f32, PIXELS_PER_THREAD>>) {
    let n_segs = fill.size_and_rule >> 1u;
    let th_ix = local_id.y * (TILE_WIDTH / PIXELS_PER_THREAD) + local_id.x;
    if th_ix < TILE_HEIGHT {
        if th_ix == 0u {
            atomicStore(&sh_winding_y[th_ix], 0u);
        }
        atomicStore(&sh_winding[th_ix], 0u);
    }
    let sample_count = PIXELS_PER_THREAD;
    for (var i = 0u; i < sample_count; i++) {
        atomicStore(&sh_samples[th_ix * sample_count + i], 0u);
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
            // Coordinates are relative to tile origin
            let xy0 = segment.point0;
            let xy1 = segment.point1;
            var y_edge_f = f32(TILE_HEIGHT);
            if xy0.x == 0.0 {
                y_edge_f = xy0.y;
            } else if xy1.x == 0.0 {
                y_edge_f = xy1.y;
            }
            // discard horizontal lines aligned to pixel grid
            if !(xy0.y == xy1.y && xy0.y == floor(xy0.y)) {
                count = span(xy0.x, xy1.x) + span(xy0.y, xy1.y) - 1u;
            }
            let y_edge = u32(ceil(y_edge_f));
            if y_edge < TILE_HEIGHT {
                atomicXor(&sh_winding_y[0], 1u << y_edge);
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
            let xy0_in = segment.point0;
            let xy1_in = segment.point1;
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
            // See comments in nonzero case.
            var is_bump = false;
            let zp = floor(a * f32(sub_ix - 1u) + b);
            if sub_ix == 0u {
                is_delta = y0i == xy0.y;
                is_bump = xy0.x == 0.0;
            } else {
                is_delta = z == zp;
                is_bump = is_positive_slope && !is_delta;
            }
            if u32(x) < TILE_WIDTH - 1u && u32(y) < TILE_HEIGHT {
                if is_delta {
                    atomicXor(&sh_winding[y], 2u << u32(x));
                }
            }
            // Apply sample mask
            let mask_block = u32(is_positive_slope) * (MASK_WIDTH * MASK_HEIGHT / 2u);
            let half_height = f32(MASK_HEIGHT / 2u);
            let mask_row = floor(min(a * half_height, half_height - 1.0)) * f32(MASK_WIDTH);
            let mask_col = floor((zf - z) * f32(MASK_WIDTH));
            let mask_ix = mask_block + u32(mask_row + mask_col);
            let pix_ix = u32(y) * TILE_WIDTH + u32(x);
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
            if is_bump {
                mask ^= 0xffu;
            }
            atomicXor(&sh_samples[pix_ix], mask);
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
            if is_bump {
                mask ^= 0xffffu;
            }
            atomicXor(&sh_samples[pix_ix], mask);
#endif
        }
        workgroupBarrier();
    }
    var area: array<f32, PIXELS_PER_THREAD>;
    var scan_x = atomicLoad(&sh_winding[local_id.y]);
    // prefix sum over GF(2) is equivalent to carry-less multiplication
    // by 0xFFFF
    scan_x ^= scan_x << 1u;
    scan_x ^= scan_x << 2u;
    scan_x ^= scan_x << 4u;
    scan_x ^= scan_x << 8u;
    // scan_x contains the winding number parity for all pixels in the row
    var scan_y = atomicLoad(&sh_winding_y[0]);
    scan_y ^= scan_y << 1u;
    scan_y ^= scan_y << 2u;
    scan_y ^= scan_y << 4u;
    scan_y ^= scan_y << 8u;
    // winding number parity for the row of pixels is in the LSB of row_parity
    let row_parity = (scan_y >> local_id.y) ^ u32(fill.backdrop);

    for (var i = 0u; i < PIXELS_PER_THREAD; i++) {
        let pix_ix = th_ix * PIXELS_PER_THREAD + i;
        let samples = atomicLoad(&sh_samples[pix_ix]);
        let pix_parity = row_parity ^ (scan_x >> (pix_ix % TILE_WIDTH));
        // The LSB of pix_parity contains the sum of the first three levels
        // of the hierarchy, thus the absolute winding number of the top left
        // of the pixel.
        let pix_mask = u32(-i32(pix_parity & 1u));
        // pix_mask is pix_party broadcast to all bits in the word.
#ifdef msaa8
        area[i] = f32(countOneBits((samples ^ pix_mask) & 0xffu)) * 0.125;
#endif
#ifdef msaa16
        area[i] = f32(countOneBits((samples ^ pix_mask) & 0xffffu)) * 0.0625;
#endif
    }
    *result = area;
}
#endif // msaa

// Error function approximation.
//
// https://raphlinus.github.io/graphics/2020/04/21/blurred-rounded-rects.html
fn erf7(x: f32) -> f32 {
    // Clamp to prevent overflow.
    // Intermediate steps calculate pow(x, 14).
    let y = clamp(x * 1.1283791671, -100.0, 100.0);
    let yy = y * y;
    let z = y + (0.24295 + (0.03395 + 0.0104 * yy) * yy) * (y * yy);
    return z / sqrt(1.0 + z * z);
}

fn hypot(a: f32, b: f32) -> f32 {
    return sqrt(a * a + b * b);
}

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

fn read_blur_rect(cmd_ix: u32) -> CmdBlurRect {
    let info_offset = ptcl[cmd_ix + 1u];
    let rgba_color = ptcl[cmd_ix + 2u];

    let m0 = bitcast<f32>(info[info_offset]);
    let m1 = bitcast<f32>(info[info_offset + 1u]);
    let m2 = bitcast<f32>(info[info_offset + 2u]);
    let m3 = bitcast<f32>(info[info_offset + 3u]);
    let matrx = vec4(m0, m1, m2, m3);
    let xlat = vec2(bitcast<f32>(info[info_offset + 4u]), bitcast<f32>(info[info_offset + 5u]));
    let width = bitcast<f32>(info[info_offset + 6u]);
    let height = bitcast<f32>(info[info_offset + 7u]);
    let radius = bitcast<f32>(info[info_offset + 8u]);
    let std_dev = bitcast<f32>(info[info_offset + 9u]);

    return CmdBlurRect(rgba_color, matrx, xlat, width, height, radius, std_dev);
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

fn read_sweep_grad(cmd_ix: u32) -> CmdSweepGrad {
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
    let t0 = bitcast<f32>(info[info_offset + 6u]);
    let t1 = bitcast<f32>(info[info_offset + 7u]);
    return CmdSweepGrad(index, extend_mode, matrx, xlat, t0, t1);
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
    let sample_alpha = info[info_offset + 8u];
    let alpha = f32(sample_alpha & 0xFFu) / 255.0;
    let format = sample_alpha >> 15u;
    let alpha_type = (sample_alpha >> 14u) & 0x1u;
    let quality = (sample_alpha >> 12u) & 0x3u;
    let x_extend = (sample_alpha >> 10u) & 0x3u;
    let y_extend = (sample_alpha >> 8u) & 0x3u;
    // The following are not intended to be bitcasts
    let x = f32(xy >> 16u);
    let y = f32(xy & 0xffffu);
    let width = f32(width_height >> 16u);
    let height = f32(width_height & 0xffffu);
    return CmdImage(matrx, xlat, vec2(x, y), vec2(width, height), format, x_extend, y_extend, quality, alpha, alpha_type);
}

fn read_end_clip(cmd_ix: u32) -> CmdEndClip {
    let blend = ptcl[cmd_ix + 1u];
    let alpha = bitcast<f32>(ptcl[cmd_ix + 2u]);
    return CmdEndClip(blend, alpha);
}

const PIXEL_FORMAT_RGBA: u32 = 0u;
const PIXEL_FORMAT_BGRA: u32 = 1u;
// Normalises subpixel order loaded from an image, based on the image's format.
fn pixel_format(pixel: vec4f, format: u32) -> vec4f {
    switch format {
        case PIXEL_FORMAT_BGRA: {
            // The conversion from RGBA to BGRA is its own inverse.
            return pixel.bgra;
        }
        case PIXEL_FORMAT_RGBA, default: {
            return pixel;
        }
    }
}

const ALPHA: u32 = 0u;
const PREMULTIPLIED_ALPHA: u32 = 1u;
// Premultiplies alpha if not already
fn maybe_premul_alpha(pixel: vec4f, alpha_type: u32) -> vec4f {
    switch alpha_type {
        case PREMULTIPLIED_ALPHA: {
            return pixel;
        }
        case ALPHA, default: {
            return premul_alpha(pixel);
        }
    }
}

const EXTEND_PAD: u32 = 0u;
const EXTEND_REPEAT: u32 = 1u;
const EXTEND_REFLECT: u32 = 2u;
fn extend_mode_normalized(t: f32, mode: u32) -> f32 {
    switch mode {
        case EXTEND_PAD: {
            return clamp(t, 0.0, 1.0);
        }
        case EXTEND_REPEAT: {
            return fract(t);
        }
        case EXTEND_REFLECT, default: {
            return abs(t - 2.0 * round(0.5 * t));
        }
    }
}

fn extend_mode(t: f32, mode: u32, max: f32) -> f32 {
    switch mode {
        case EXTEND_PAD: {
            return clamp(t, 0.0, max);
        }
        case EXTEND_REPEAT: {
            return extend_mode_normalized(t / max, mode) * max;
        }
        case EXTEND_REFLECT, default: {
            return extend_mode_normalized(t / max, mode) * max;
        }
    }
}

const PIXELS_PER_THREAD = 4u;

#ifndef msaa

// Analytic area anti-aliasing.
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
        let y = segment.point0.y - xy.y;
        let delta = segment.point1 - segment.point0;
        let y0 = clamp(y, 0.0, 1.0);
        let y1 = clamp(y + delta.y, 0.0, 1.0);
        let dy = y0 - y1;
        if dy != 0.0 {
            let vec_y_recip = 1.0 / delta.y;
            let t0 = (y0 - y) * vec_y_recip;
            let t1 = (y1 - y) * vec_y_recip;
            let startx = segment.point0.x - xy.x;
            let x0 = startx + t0 * delta.x;
            let x1 = startx + t1 * delta.x;
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
        let y_edge = sign(delta.x) * clamp(xy.y - segment.y_edge + 1.0, 0.0, 1.0);
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

#endif

// The X size should be 16 / PIXELS_PER_THREAD
@compute @workgroup_size(4, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    if ptcl[0] == ~0u {
        // An earlier stage has failed, don't try to render.
        // We use ptcl[0] for this so we don't use up a binding for bump.
        return;
    }
    let tile_ix = wg_id.y * config.width_in_tiles + wg_id.x;
    let xy = vec2(f32(global_id.x * PIXELS_PER_THREAD), f32(global_id.y));
    let local_xy = vec2(f32(local_id.x * PIXELS_PER_THREAD), f32(local_id.y));
    var rgba: array<vec4<f32>, PIXELS_PER_THREAD>;
    let base_color = unpack4x8unorm(config.base_color);
    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
        rgba[i] = base_color;
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
            case CMD_FILL: {
                let fill = read_fill(cmd_ix);
#ifdef msaa
                fill_path_ms(fill, local_id.xy, &area);
#else
                fill_path(fill, local_xy, &area);
#endif
                cmd_ix += 4u;
            }
            case CMD_SOLID: {
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    area[i] = 1.0;
                }
                cmd_ix += 1u;
            }
            case CMD_COLOR: {
                let color = read_color(cmd_ix);
                let fg = unpack4x8unorm(color.rgba_color);
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    let fg_i = fg * area[i];
                    rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                }
                cmd_ix += 2u;
            }
            case CMD_BEGIN_CLIP: {
                if clip_depth < BLEND_STACK_SPLIT {
                    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                        blend_stack[clip_depth][i] = pack4x8unorm(rgba[i]);
                        rgba[i] = vec4(0.0);
                    }
                } else {
                    let blend_in_scratch = clip_depth - BLEND_STACK_SPLIT;
                    let local_tile_ix = local_id.x * PIXELS_PER_THREAD + local_id.y * TILE_WIDTH;
                    let local_blend_start = blend_offset + blend_in_scratch * TILE_WIDTH * TILE_HEIGHT + local_tile_ix;
                    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                        blend_spill[local_blend_start + i] = pack4x8unorm(rgba[i]);
                        rgba[i] = vec4(0.0);
                    }
                }
                clip_depth += 1u;
                cmd_ix += 1u;
            }
            case CMD_END_CLIP: {
                let end_clip = read_end_clip(cmd_ix);
                clip_depth -= 1u;
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    var bg_rgba: u32;
                    if clip_depth < BLEND_STACK_SPLIT {
                        bg_rgba = blend_stack[clip_depth][i];
                    } else {
                        let blend_in_scratch = clip_depth - BLEND_STACK_SPLIT;
                        let local_tile_ix = local_id.x * PIXELS_PER_THREAD + local_id.y * TILE_WIDTH;
                        let local_blend_start = blend_offset + blend_in_scratch * TILE_WIDTH * TILE_HEIGHT + local_tile_ix;
                        bg_rgba = blend_spill[local_blend_start + i];
                    }
                    let bg = unpack4x8unorm(bg_rgba);
                    let fg = rgba[i] * area[i] * end_clip.alpha;
                    if end_clip.blend == LUMINANCE_MASK_LAYER {
                        // TODO: Does this case apply more generally?
                        // See https://github.com/linebender/vello/issues/1061
                        // TODO: How do we handle anti-aliased edges here?
                        // This is really an imaging model question
                        if area[i] == 0f {
                            rgba[i] = bg;
                            continue;
                        }
                        let luminance = clamp(svg_lum(unpremultiply(fg)) * fg.a, 0.0, 1.0);
                        rgba[i] = bg * luminance;
                    } else {
                        rgba[i] = blend_mix_compose(bg, fg, end_clip.blend);
                    }
                }
                cmd_ix += 3u;
            }
            case CMD_JUMP: {
                cmd_ix = ptcl[cmd_ix + 1u];
            }
            case CMD_BLUR_RECT: {
                /// Approximation for the convolution of a gaussian filter with a rounded rectangle.
                ///
                /// See https://raphlinus.github.io/graphics/2020/04/21/blurred-rounded-rects.html

                let blur = read_blur_rect(cmd_ix);

                // Avoid division by 0
                let std_dev = max(blur.std_dev, 1e-5);
                let inv_std_dev = 1.0 / std_dev;
                
                let min_edge = min(blur.width, blur.height);
                let radius_max = 0.5 * min_edge;
                let r0 = min(hypot(blur.radius, std_dev * 1.15), radius_max);
                let r1 = min(hypot(blur.radius, std_dev * 2.0), radius_max);

                let exponent = 2.0 * r1 / r0;
                let inv_exponent = 1.0 / exponent;
                
                // Pull in long end (make less eccentric).
                let delta = 1.25 * std_dev * (exp(-pow(0.5 * inv_std_dev * blur.width, 2.0)) - exp(-pow(0.5 * inv_std_dev * blur.height, 2.0)));
                let width = blur.width + min(delta, 0.0);
                let height = blur.height - max(delta, 0.0);

                let scale = 0.5 * erf7(inv_std_dev * 0.5 * (max(width, height) - 0.5 * blur.radius));

                let blur_rgba = unpack4x8unorm(blur.rgba_color);

                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    // Transform fragment location to local 'uv' space of the rounded rectangle.
                    let my_xy = vec2(xy.x + f32(i), xy.y);
                    let local_xy = blur.matrx.xy * my_xy.x + blur.matrx.zw * my_xy.y + blur.xlat;
                    let x = local_xy.x;
                    let y = local_xy.y;

                    let y0 = abs(y) - (height * 0.5 - r1);
                    let y1 = max(y0, 0.0);

                    let x0 = abs(x) - (width * 0.5 - r1);
                    let x1 = max(x0, 0.0);

                    let d_pos = pow(pow(x1, exponent) + pow(y1, exponent), inv_exponent);
                    let d_neg = min(max(x0, y0), 0.0);
                    let d = d_pos + d_neg - r1;
                    let alpha = scale * (erf7(inv_std_dev * (min_edge + d)) - erf7(inv_std_dev * d));

                    let fg_rgba = blur_rgba * alpha;
                    let fg_i = fg_rgba * area[i];
                    rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                }
                cmd_ix += 3u;
            }
            case CMD_LIN_GRAD: {
                let lin = read_lin_grad(cmd_ix);
                let d = lin.line_x * xy.x + lin.line_y * xy.y + lin.line_c;
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    let my_d = d + lin.line_x * f32(i);
                    let x = i32(round(extend_mode_normalized(my_d, lin.extend_mode) * f32(GRADIENT_WIDTH - 1)));
                    let fg_rgba = textureLoad(gradients, vec2(x, i32(lin.index)), 0);
                    let fg_i = fg_rgba * area[i];
                    rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                }
                cmd_ix += 3u;
            }
            case CMD_RAD_GRAD: {
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
                        t = extend_mode_normalized(focal_x + t_sign * t, rad.extend_mode);
                        t = select(t, 1.0 - t, is_swapped);
                        let x = i32(round(t * f32(GRADIENT_WIDTH - 1)));
                        let fg_rgba = textureLoad(gradients, vec2(x, i32(rad.index)), 0);
                        let fg_i = fg_rgba * area[i];
                        rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                    }
                }
                cmd_ix += 3u;
            }
            case CMD_SWEEP_GRAD: {
                let sweep = read_sweep_grad(cmd_ix);
                let scale = 1.0 / (sweep.t1 - sweep.t0);
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    let my_xy = vec2(xy.x + f32(i), xy.y);
                    let local_xy = sweep.matrx.xy * my_xy.x + sweep.matrx.zw * my_xy.y + sweep.xlat;
                    let x = local_xy.x;
                    let y = local_xy.y;
                    // xy_to_unit_angle from Skia:
                    // See <https://github.com/google/skia/blob/30bba741989865c157c7a997a0caebe94921276b/src/opts/SkRasterPipeline_opts.h#L5859>
                    let xabs = abs(x);
                    let yabs = abs(y);
                    let slope = min(xabs, yabs) / max(xabs, yabs);
                    let s = slope * slope;
                    // again, from Skia:
                    // Use a 7th degree polynomial to approximate atan.
                    // This was generated using sollya.gforge.inria.fr.
                    // A float optimized polynomial was generated using the following command.
                    // P1 = fpminimax((1/(2*Pi))*atan(x),[|1,3,5,7|],[|24...|],[2^(-40),1],relative);
                    var phi = slope * (0.15912117063999176025390625f + s * (-5.185396969318389892578125e-2f + s * (2.476101927459239959716796875e-2f + s * (-7.0547382347285747528076171875e-3f))));
                    phi = select(phi, 1.0 / 4.0 - phi, xabs < yabs);
                    phi = select(phi, 1.0 / 2.0 - phi, x < 0.0);
                    phi = select(phi, 1.0 - phi, y < 0.0);
                    phi = select(phi, 0.0, phi != phi); // check for NaN
                    phi = (phi - sweep.t0) * scale;
                    let t = extend_mode_normalized(phi, sweep.extend_mode);
                    let ramp_x = i32(round(t * f32(GRADIENT_WIDTH - 1)));
                    let fg_rgba = textureLoad(gradients, vec2(ramp_x, i32(sweep.index)), 0);
                    let fg_i = fg_rgba * area[i];
                    rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                }
                cmd_ix += 3u;
            }
            case CMD_IMAGE: {
                let image = read_image(cmd_ix);
                let atlas_max = image.atlas_offset + image.extents - vec2(1.0);
                switch image.quality {
                    case IMAGE_QUALITY_LOW: {
                        for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                            // We only need to load from the textures if the value will be used.
                            if area[i] != 0.0 {
                                let my_xy = vec2(xy.x + f32(i), xy.y);
                                var atlas_uv = image.matrx.xy * my_xy.x + image.matrx.zw * my_xy.y + image.xlat;
                                atlas_uv.x = extend_mode(atlas_uv.x, image.x_extend_mode, image.extents.x);
                                atlas_uv.y = extend_mode(atlas_uv.y, image.y_extend_mode, image.extents.y);
                                atlas_uv = atlas_uv + image.atlas_offset;
                                // TODO: If the image couldn't be added to the atlas (i.e. was too big), this isn't robust
                                let atlas_uv_clamped = clamp(atlas_uv, image.atlas_offset, atlas_max);
                                // Nearest neighbor sampling
                                let fg_rgba = maybe_premul_alpha(textureLoad(image_atlas, vec2<i32>(atlas_uv_clamped), 0), image.alpha_type);
                                let fg_i = pixel_format(fg_rgba * area[i] * image.alpha, image.format);
                                rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                            }
                        }
                    }
                    case IMAGE_QUALITY_MEDIUM, default: {
                        // We don't have an implementation for `IMAGE_QUALITY_HIGH` yet, just use the same as medium
                        for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                            // We only need to load from the textures if the value will be used.
                            if area[i] != 0.0 {
                                let my_xy = vec2(xy.x + f32(i), xy.y);
                                var atlas_uv = image.matrx.xy * my_xy.x + image.matrx.zw * my_xy.y + image.xlat;
                                atlas_uv.x = extend_mode(atlas_uv.x, image.x_extend_mode, image.extents.x);
                                atlas_uv.y = extend_mode(atlas_uv.y, image.y_extend_mode, image.extents.y);
                                atlas_uv = atlas_uv + image.atlas_offset - vec2(0.5);
                                // TODO: If the image couldn't be added to the atlas (i.e. was too big), this isn't robust
                                let atlas_uv_clamped = clamp(atlas_uv, image.atlas_offset, atlas_max);
                                // We know that the floor and ceil are within the atlas area because atlas_max and
                                // atlas_offset are integers
                                let uv_quad = vec4(floor(atlas_uv_clamped), ceil(atlas_uv_clamped));
                                let uv_frac = fract(atlas_uv);
                                let a = maybe_premul_alpha(textureLoad(image_atlas, vec2<i32>(uv_quad.xy), 0), image.alpha_type);
                                let b = maybe_premul_alpha(textureLoad(image_atlas, vec2<i32>(uv_quad.xw), 0), image.alpha_type);
                                let c = maybe_premul_alpha(textureLoad(image_atlas, vec2<i32>(uv_quad.zy), 0), image.alpha_type);
                                let d = maybe_premul_alpha(textureLoad(image_atlas, vec2<i32>(uv_quad.zw), 0), image.alpha_type);
                                // Bilinear sampling
                                let fg_rgba = mix(mix(a, b, uv_frac.y), mix(c, d, uv_frac.y), uv_frac.x);
                                let fg_i = pixel_format(fg_rgba * area[i] * image.alpha, image.format);
                                rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                            }
                        }
                    }
                }
                cmd_ix += 2u;
            }
            default: {}
        }
    }
    let xy_uint = vec2<u32>(xy);
    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
        let coords = xy_uint + vec2(i, 0u);
        if coords.x < config.target_width && coords.y < config.target_height {
            let fg = rgba[i];
            // let fg = base_color * (1.0 - foreground.a) + foreground;
            // Max with a small epsilon to avoid NaNs
            let a_inv = 1.0 / max(fg.a, 1e-6);
            let rgba_sep = vec4(fg.rgb * a_inv, fg.a);
            textureStore(output, vec2<i32>(coords), rgba_sep);
        }
    } 
}

fn premul_alpha(rgba: vec4<f32>) -> vec4<f32> {
    return vec4(rgba.rgb * rgba.a, rgba.a);
}
