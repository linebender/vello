// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// The texture that holds the encoded parameters for all filter effects used in the scene.
@group(0) @binding(0) var filter_data: texture_2d<u32>;
// The texture holding the input texture we want to filter.
@group(1) @binding(0) var in_tex: texture_2d<f32>;
// A bilinear sampler.
@group(1) @binding(1) var linear_sampler: sampler;
// The texture containing the original (unfiltered) content. This is only needed because
// for the drop shadow filter, we need to composite the original content on top of the shadow.
@group(2) @binding(0) var original_tex: texture_2d<f32>;

// Keep these variables and structs in sync with the ones in `filter.rs`!

const FILTER_SIZE_BYTES: u32 = 48;
const FILTER_SIZE_U32: u32 = FILTER_SIZE_BYTES / 4;
const TEXELS_PER_FILTER: u32 = FILTER_SIZE_U32 / 4u;

const FILTER_TYPE_OFFSET: u32 = 0u;
const FILTER_TYPE_FLOOD: u32 = 1u;
const FILTER_TYPE_GAUSSIAN_BLUR: u32 = 2u;
const FILTER_TYPE_DROP_SHADOW: u32 = 3u;

const PASS_COPY: u32 = 0u;
const PASS_FLOOD: u32 = 1u;
const PASS_OFFSET: u32 = 2u;
const PASS_DOWNSCALE: u32 = 3u;
const PASS_BLUR_H: u32 = 4u;
const PASS_BLUR_V: u32 = 5u;
const PASS_UPSCALE: u32 = 6u;
const PASS_COMPOSITE: u32 = 7u;

const MAX_TAPS_PER_SIDE: u32 = 3u;

// A type erased instance of a filter containing the values of all parameters.
struct GpuFilterData {
    data: array<u32, 12>, // 12 = FILTER_SIZE_U32
}

struct OffsetFilter {
    dx: f32,
    dy: f32,
}

struct FloodFilter {
    color: u32,
}

struct BlurParams {
    n_linear_taps: u32,
    center_weight: f32,
    linear_weights: array<f32, MAX_TAPS_PER_SIDE>,
    linear_offsets: array<f32, MAX_TAPS_PER_SIDE>,
}

struct DropShadowFilter {
    dx: f32,
    dy: f32,
    color: u32,
}

// The layout of the header:
//   bits [0:4]   = filter_type   (5 bits)
//   bits [5:6]   = edge_mode     (2 bits, only for blur filters), currently ignored.
//   bits [7:10]  = n_decimations (4 bits, only for blur filters), only read on the CPU side.
//   bits [11:12] = n_linear_taps (2 bits, only for blur filters)
//   bits [13:32] = reserved for future use

fn unpack_filter_type(data: GpuFilterData) -> u32 { return data.data[0] & 0x1Fu; }
fn unpack_header_n_linear_taps(header: u32) -> u32 { return (header >> 11u) & 0x3u; }

fn unpack_offset_filter(data: GpuFilterData) -> OffsetFilter {
    return OffsetFilter(
        bitcast<f32>(data.data[1]),
        bitcast<f32>(data.data[2])
    );
}

fn unpack_flood_filter(data: GpuFilterData) -> FloodFilter {
    return FloodFilter(data.data[1]);
}

// Note that this assumes that the data is stored directly after the header,
// which currently is the case for gaussian blur and drop shadow.
fn unpack_blur_params(data: GpuFilterData) -> BlurParams {
    let n_linear_taps = unpack_header_n_linear_taps(data.data[0]);
    let center_weight = bitcast<f32>(data.data[1]);
    var weights: array<f32, MAX_TAPS_PER_SIDE>;
    var offsets: array<f32, MAX_TAPS_PER_SIDE>;

    for (var i = 0u; i < MAX_TAPS_PER_SIDE; i++) {
        weights[i] = bitcast<f32>(data.data[2u + i]);
        offsets[i] = bitcast<f32>(data.data[2u + MAX_TAPS_PER_SIDE + i]);
    }

    return BlurParams(n_linear_taps, center_weight, weights, offsets);
}

fn unpack_drop_shadow_filter(data: GpuFilterData) -> DropShadowFilter {
    return DropShadowFilter(
        bitcast<f32>(data.data[8]),
        bitcast<f32>(data.data[9]),
        data.data[10],
    );
}

fn load_filter_data(texel_offset: u32) -> GpuFilterData {
    let w = textureDimensions(filter_data).x;
    let t0 = textureLoad(filter_data, vec2((texel_offset     ) % w, (texel_offset     ) / w), 0);
    let t1 = textureLoad(filter_data, vec2((texel_offset + 1u) % w, (texel_offset + 1u) / w), 0);
    let t2 = textureLoad(filter_data, vec2((texel_offset + 2u) % w, (texel_offset + 2u) / w), 0);
    return GpuFilterData(array(t0.x, t0.y, t0.z, t0.w, t1.x, t1.y, t1.z, t1.w, t2.x, t2.y, t2.z, t2.w));
}

fn unpack_color(packed: u32) -> vec4<f32> {
    return unpack4x8unorm(packed);
}

struct FilterInstanceData {
    @location(0) src_offset: vec2<u32>,
    @location(1) src_size: vec2<u32>,
    @location(2) dest_offset: vec2<u32>,
    @location(3) dest_size: vec2<u32>,
    @location(4) dest_atlas_size: vec2<u32>,
    @location(5) filter_offset: u32,
    @location(6) original_offset: vec2<u32>,
    @location(7) original_size: vec2<u32>,
    @location(8) pass_kind: u32,
}

struct FilterVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) filter_offset: u32,
    @location(1) @interpolate(flat) src_offset: vec2<u32>,
    @location(2) @interpolate(flat) src_size: vec2<u32>,
    @location(3) @interpolate(flat) dest_offset: vec2<u32>,
    @location(4) @interpolate(flat) dest_size: vec2<u32>,
    @location(5) @interpolate(flat) dest_atlas_size: vec2<u32>,
    @location(6) @interpolate(flat) original_offset: vec2<u32>,
    @location(7) @interpolate(flat) original_size: vec2<u32>,
    @location(8) @interpolate(flat) pass_kind: u32,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    instance: FilterInstanceData
) -> FilterVertexOutput {
    // Generate quad (0-3) covering the dest region
    let quad_vertex = vertex_index % 4u;
    let x = f32((quad_vertex & 1u));      // 0,1,0,1
    let y = f32((quad_vertex >> 1u));      // 0,0,1,1

    // Calculate pixel position in atlas. The quad covers the full allocation size
    // (original_size), which may be larger than dest_size after decimation.
    // The fragment shader returns transparent for pixels outside dest_size.
    let pix_x = f32(instance.dest_offset.x) + x * f32(instance.original_size.x);
    let pix_y = f32(instance.dest_offset.y) + y * f32(instance.original_size.y);

    // Convert to NDC using the dest atlas dimensions
    let atlas_size = vec2<f32>(instance.dest_atlas_size);
    let ndc_x = pix_x * 2.0 / atlas_size.x - 1.0;
    let ndc_y = 1.0 - pix_y * 2.0 / atlas_size.y;

    var out: FilterVertexOutput;
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.filter_offset = instance.filter_offset;
    out.src_offset = instance.src_offset;
    out.src_size = instance.src_size;
    out.dest_offset = instance.dest_offset;
    out.dest_size = instance.dest_size;
    out.dest_atlas_size = instance.dest_atlas_size;
    out.original_offset = instance.original_offset;
    out.original_size = instance.original_size;
    out.pass_kind = instance.pass_kind;
    return out;
}

// Atlas padding around filter images guarantees transparent reads beyond the content
// region, so these sampling functions omit bounds checks. The one exception is OFFSET,
// which can shift by large dx/dy (e.g. 50 px for drop shadow) exceeding the padding;
// that path uses sample_source_checked below.

fn sample_original(in: FilterVertexOutput, rel_coord: vec2<f32>) -> vec4<f32> {
    let src_coord = vec2<u32>(vec2<i32>(in.original_offset) + vec2<i32>(rel_coord));
    return textureLoad(original_tex, src_coord, 0);
}

fn sample_source(in: FilterVertexOutput, rel_coord: vec2<f32>) -> vec4<f32> {
    let src_coord = vec2<u32>(vec2<i32>(in.src_offset) + vec2<i32>(rel_coord));
    return textureLoad(in_tex, src_coord, 0);
}

// Bounds-checked variant of sample_source, used only by OFFSET passes where the
// shift can exceed the atlas padding.
fn sample_source_checked(in: FilterVertexOutput, rel_coord: vec2<f32>) -> vec4<f32> {
    if rel_coord.x < 0.0 || rel_coord.x >= f32(in.src_size.x) ||
       rel_coord.y < 0.0 || rel_coord.y >= f32(in.src_size.y) {
        return vec4<f32>(0.0);
    }
    let src_coord = vec2<u32>(vec2<i32>(in.src_offset) + vec2<i32>(rel_coord));
    return textureLoad(in_tex, src_coord, 0);
}

// TODO: Add support for edge modes when blurring. This unfortunately will make it harder/impossible to perform
// filtering with bilinear samplers (since the padding of images is always transparent), hence why this is currently
// not implemented. It's possible, but we should only do it if we really need it.

// 2x decimation using [1,3,3,1]x[1,3,3,1]/64 binomial filter with bilinear sampling.
//
// The 1D [1,3,3,1]/8 kernel at offsets {-1, 0, +1, +2} relative to src_center is
// merged into two bilinear taps:
//   pair(-1, 0): weight 1/8+3/8 = 0.5, offset (-1*1 + 0*3)/4 = -0.25
//   pair(+1,+2): weight 3/8+1/8 = 0.5, offset (+1*3 + 2*1)/4 =  1.25
// In 2D all four combinations have equal weight 0.25, so the result is the average
// of four bilinear samples. This reduces 16 textureLoad calls to 4 textureSample calls.
fn decimate_filter(in: FilterVertexOutput) -> vec4<f32> {
    let frag_coord = vec2<u32>(in.position.xy);
    let rel = vec2<i32>(frag_coord - in.dest_offset);
    let src_center = vec2<f32>(rel * 2);
    let atlas_center = vec2<f32>(in.src_offset) + src_center;
    let tex_size = vec2<f32>(textureDimensions(in_tex));

    // Bilinear merge offsets for the [1,3,3,1] kernel pairs.
    let lo = vec2<f32>(-0.25);  // pair(-1, 0)
    let hi = vec2<f32>( 1.25);  // pair(+1, +2)

    let s00 = textureSample(in_tex, linear_sampler, (atlas_center + vec2(lo.x, lo.y) + 0.5) / tex_size);
    let s01 = textureSample(in_tex, linear_sampler, (atlas_center + vec2(lo.x, hi.y) + 0.5) / tex_size);
    let s10 = textureSample(in_tex, linear_sampler, (atlas_center + vec2(hi.x, lo.y) + 0.5) / tex_size);
    let s11 = textureSample(in_tex, linear_sampler, (atlas_center + vec2(hi.x, hi.y) + 0.5) / tex_size);

    return (s00 + s01 + s10 + s11) * 0.25;
}

// 2x upscale with bilinear sampling.
//
// Each output pixel maps to a source position at (base + phase * 0.5) where phase is
// 0 or 1 per axis. The [0.75, 0.25] weighting between the main pixel and its neighbor
// is exactly what GPU bilinear interpolation produces when the sample point is offset
// by +/-0.25 from the pixel center. This reduces 4 textureLoad calls to 1 textureSample.
fn upscale_filter(in: FilterVertexOutput) -> vec4<f32> {
    let frag_coord = vec2<u32>(in.position.xy);
    let rel = vec2<i32>(frag_coord - in.dest_offset);
    let src_base = vec2<f32>(rel / 2);
    let phase = vec2<f32>(rel % 2);
    let tex_size = vec2<f32>(textureDimensions(in_tex));

    // phase=0 -> sample at base - 0.25 (0.75 main + 0.25 left/top neighbor)
    // phase=1 -> sample at base + 0.25 (0.75 main + 0.25 right/bottom neighbor)
    let sample_offset = select(vec2(-0.25), vec2(0.25), phase == vec2(1.0));
    let atlas_pos = vec2<f32>(in.src_offset) + src_base + sample_offset + 0.5;

    return textureSample(in_tex, linear_sampler, atlas_pos / tex_size);
}

fn convolve(
    in: FilterVertexOutput,
    center_in: vec2<f32>,
    dir: vec2<f32>,
    n_linear_taps: u32,
    center_weight: f32,
    weights: array<f32, 3>,
    offsets: array<f32, 3>,
) -> vec4<f32> {
    let atlas_center = vec2<f32>(in.src_offset) + center_in;
    let tex_size = vec2<f32>(textureDimensions(in_tex));

    // Atlas padding guarantees transparent reads beyond content, so we can always
    // use the fast GPU-native linear sampling path without bounds checking.
    let center_uv = (atlas_center + 0.5) / tex_size;
    var color = textureSample(in_tex, linear_sampler, center_uv) * center_weight;
    for (var i = 0u; i < n_linear_taps; i++) {
        let w = weights[i];
        let d = dir * offsets[i];
        let pos_uv = (atlas_center + d + 0.5) / tex_size;
        let neg_uv = (atlas_center - d + 0.5) / tex_size;
        color += textureSample(in_tex, linear_sampler, pos_uv) * w;
        color += textureSample(in_tex, linear_sampler, neg_uv) * w;
    }
    return color;
}

const HORIZONTAL: vec2<f32> = vec2<f32>(1.0, 0.0);
const VERTICAL: vec2<f32> = vec2<f32>(0.0, 1.0);

@fragment
fn fs_main(in: FilterVertexOutput) -> @location(0) vec4<f32> {
    let frag_coord = vec2<u32>(in.position.xy);
    let rel_coord = vec2<f32>(frag_coord - in.dest_offset);

    // The vertex quad covers the full allocation size, but the actual dest region
    // may be smaller (e.g. after decimation). Return transparent for OOB pixels.
    if rel_coord.x >= f32(in.dest_size.x) || rel_coord.y >= f32(in.dest_size.y) {
        return vec4<f32>(0.0);
    }

    let data = load_filter_data(in.filter_offset);

    switch in.pass_kind {
        case PASS_COPY: {
            return sample_source(in, rel_coord);
        }
        case PASS_FLOOD: {
            let flood = unpack_flood_filter(data);
            return unpack_color(flood.color);
        }
        case PASS_OFFSET: {
            let filter_type = unpack_filter_type(data);
            if filter_type == FILTER_TYPE_DROP_SHADOW {
                let shadow = unpack_drop_shadow_filter(data);
                // For drop shadow, use floor to match the old combined offset+blur behavior.
                return sample_source_checked(in, floor(rel_coord - vec2<f32>(shadow.dx, shadow.dy)));
            } else {
                let offset = unpack_offset_filter(data);
                return sample_source_checked(in, round(rel_coord - vec2<f32>(offset.dx, offset.dy)));
            }
        }
        case PASS_DOWNSCALE: {
            return decimate_filter(in);
        }
        case PASS_BLUR_H: {
            let blur = unpack_blur_params(data);
            return convolve(in, rel_coord, HORIZONTAL, blur.n_linear_taps, blur.center_weight, blur.linear_weights, blur.linear_offsets);
        }
        case PASS_BLUR_V: {
            let blur = unpack_blur_params(data);
            return convolve(in, rel_coord, VERTICAL, blur.n_linear_taps, blur.center_weight, blur.linear_weights, blur.linear_offsets);
        }
        case PASS_UPSCALE: {
            return upscale_filter(in);
        }
        case PASS_COMPOSITE: {
            // Drop shadow composite: colorize blurred result, composite original on top.
            let shadow = unpack_drop_shadow_filter(data);
            let blurred = sample_source(in, rel_coord);
            let shadow_color = unpack_color(shadow.color);
            let shadow_result = shadow_color * blurred.a;
            let original = sample_original(in, rel_coord);
            return original + shadow_result * (1.0 - original.a);
        }
        default: {
            return vec4<f32>(0.0);
        }
    }
}
