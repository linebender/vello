// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// The texture that holds the encoded parameters for all filter effects used in the scene.
@group(0) @binding(0) var filter_data: texture_2d<u32>;
// The texture holding the input texture we want to filter.
@group(1) @binding(0) var in_tex: texture_2d<f32>;
// Linear sampler used for more efficient sampling during Gaussian blur.
@group(1) @binding(1) var linear_sampler: sampler;
// The original (unfiltered) content texture, used by COMPOSITE passes for drop shadow.
// For non-COMPOSITE passes, this is bound to the same texture as in_tex (harmless).
@group(2) @binding(0) var original_tex: texture_2d<f32>;

// Keep these variables and structs in sync with the ones in `filter.rs`!

const FILTER_SIZE_U32: u32 = 12u;
// Since the texture is packed into Uint32.
const TEXELS_PER_FILTER: u32 = FILTER_SIZE_U32 / 4u;

const FILTER_TYPE_OFFSET: u32 = 0u;
const FILTER_TYPE_FLOOD: u32 = 1u;
const FILTER_TYPE_GAUSSIAN_BLUR: u32 = 2u;
const FILTER_TYPE_DROP_SHADOW: u32 = 3u;

// Pass kind constants (keep in sync with pass_kind module in filter.rs).
const PASS_COPY: u32 = 0u;
const PASS_FLOOD: u32 = 1u;
const PASS_OFFSET: u32 = 2u;
const PASS_DOWNSCALE: u32 = 3u;
const PASS_BLUR_H: u32 = 4u;
const PASS_BLUR_V: u32 = 5u;
const PASS_UPSCALE: u32 = 6u;
const PASS_COMPOSITE: u32 = 7u;

const MAX_LINEAR_TAPS: u32 = 3u;

struct GpuFilterData {
    data: array<u32, 12>,
}

struct OffsetFilter {
    dx: f32,
    dy: f32,
}

struct FloodFilter {
    color: u32,
}

// TODO: Support edge modes

// Header packing layout (data[0]):
//   bits [0:4]   = filter_type   (5 bits)
//   bits [5:6]   = edge_mode     (2 bits)
//   bits [7:10]  = n_decimations (4 bits)
//   bits [11:12] = n_linear_taps (2 bits)

// Blur parameters extracted from either GaussianBlur or DropShadow filter data.
struct BlurParams {
    n_linear_taps: u32,
    center_weight: f32,
    linear_weights: array<f32, MAX_LINEAR_TAPS>,
    linear_offsets: array<f32, MAX_LINEAR_TAPS>,
}

struct DropShadowFilter {
    dx: f32,
    dy: f32,
    color: u32,
    std_deviation: f32,
    edge_mode: u32,
    n_decimations: u32,
    n_linear_taps: u32,
    center_weight: f32,
    linear_weights: array<f32, MAX_LINEAR_TAPS>,
    linear_offsets: array<f32, MAX_LINEAR_TAPS>,
}

fn get_filter_type(data: GpuFilterData) -> u32 {
    return data.data[0] & 0x1Fu;
}

fn unpack_header_edge_mode(header: u32) -> u32 { return (header >> 5u) & 0x3u; }
fn unpack_header_n_decimations(header: u32) -> u32 { return (header >> 7u) & 0xFu; }
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

// Extract blur convolution parameters from filter data, regardless of whether
// the underlying filter is GaussianBlur or DropShadow.
fn unpack_blur_params(data: GpuFilterData) -> BlurParams {
    let header = data.data[0];
    let filter_type = header & 0x1Fu;
    let n_linear_taps = unpack_header_n_linear_taps(header);

    var center_weight: f32;
    var weights: array<f32, 3>;
    var offsets: array<f32, 3>;

    if filter_type == FILTER_TYPE_GAUSSIAN_BLUR {
        // GaussianBlur layout: [header, std_dev, center_weight, w0, w1, w2, o0, o1, o2, ...]
        center_weight = bitcast<f32>(data.data[2]);
        for (var i = 0u; i < 3u; i++) {
            weights[i] = bitcast<f32>(data.data[3u + i]);
            offsets[i] = bitcast<f32>(data.data[6u + i]);
        }
    } else {
        // DropShadow layout: [header, dx, dy, color, std_dev, center_weight, w0, w1, w2, o0, o1, o2]
        center_weight = bitcast<f32>(data.data[5]);
        for (var i = 0u; i < 3u; i++) {
            weights[i] = bitcast<f32>(data.data[6u + i]);
            offsets[i] = bitcast<f32>(data.data[9u + i]);
        }
    }

    return BlurParams(n_linear_taps, center_weight, weights, offsets);
}

fn unpack_drop_shadow_filter(data: GpuFilterData) -> DropShadowFilter {
    let header = data.data[0];
    var weights: array<f32, 3>;
    var offsets: array<f32, 3>;
    for (var i = 0u; i < 3u; i++) {
        weights[i] = bitcast<f32>(data.data[6u + i]);
        offsets[i] = bitcast<f32>(data.data[9u + i]);
    }
    return DropShadowFilter(
        bitcast<f32>(data.data[1]),              // dx
        bitcast<f32>(data.data[2]),              // dy
        data.data[3],                            // color
        bitcast<f32>(data.data[4]),              // std_deviation
        unpack_header_edge_mode(header),         // edge_mode
        unpack_header_n_decimations(header),     // n_decimations
        unpack_header_n_linear_taps(header),     // n_linear_taps
        bitcast<f32>(data.data[5]),              // center_weight
        weights,
        offsets,
    );
}

fn load_filter_data(texel_offset: u32) -> GpuFilterData {
    let tex_width = textureDimensions(filter_data).x;
    var data: GpuFilterData;
    for (var i = 0u; i < TEXELS_PER_FILTER; i++) {
        let idx = texel_offset + i;
        let x = idx % tex_width;
        let y = idx / tex_width;
        let texel = textureLoad(filter_data, vec2<u32>(x, y), 0);
        data.data[i * 4u + 0u] = texel.x;
        data.data[i * 4u + 1u] = texel.y;
        data.data[i * 4u + 2u] = texel.z;
        data.data[i * 4u + 3u] = texel.w;
    }
    return data;
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

    // Calculate pixel position in atlas
    let pix_x = f32(instance.dest_offset.x) + x * f32(instance.dest_size.x);
    let pix_y = f32(instance.dest_offset.y) + y * f32(instance.dest_size.y);

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

fn sample_original(in: FilterVertexOutput, rel_coord: vec2<f32>) -> vec4<f32> {
    if rel_coord.x < 0.0 || rel_coord.x >= f32(in.original_size.x) ||
       rel_coord.y < 0.0 || rel_coord.y >= f32(in.original_size.y) {
        return vec4<f32>(0.0);
    }
    let src_coord = vec2<u32>(vec2<i32>(in.original_offset) + vec2<i32>(rel_coord));
    return textureLoad(original_tex, src_coord, 0);
}

fn sample_source(in: FilterVertexOutput, rel_coord: vec2<f32>) -> vec4<f32> {
    if rel_coord.x < 0.0 || rel_coord.x >= f32(in.src_size.x) ||
       rel_coord.y < 0.0 || rel_coord.y >= f32(in.src_size.y) {
        return vec4<f32>(0.0);
    }
    let src_coord = vec2<u32>(vec2<i32>(in.src_offset) + vec2<i32>(rel_coord));
    return textureLoad(in_tex, src_coord, 0);
}

// Sample input for downscale/upscale with EdgeMode::None (transparent for out-of-bounds).
fn sample_input_edge_none(in: FilterVertexOutput, coord: vec2<i32>) -> vec4<f32> {
    if coord.x < 0 || coord.x >= i32(in.src_size.x) ||
       coord.y < 0 || coord.y >= i32(in.src_size.y) {
        return vec4<f32>(0.0);
    }
    let src_coord = vec2<u32>(coord) + in.src_offset;
    return textureLoad(in_tex, src_coord, 0);
}

// 1D weight for [1,3,3,1]/8 binomial kernel.
fn binomial_weight(i: u32) -> f32 {
    switch i {
        case 0u: { return 1.0 / 8.0; }
        case 1u: { return 3.0 / 8.0; }
        case 2u: { return 3.0 / 8.0; }
        case 3u: { return 1.0 / 8.0; }
        default: { return 0.0; }
    }
}

// 2x decimation using [1,3,3,1]x[1,3,3,1]/64 binomial filter.
fn decimate_filter(in: FilterVertexOutput) -> vec4<f32> {
    let frag_coord = vec2<u32>(in.position.xy);
    let rel = vec2<i32>(frag_coord - in.dest_offset);
    let src_center = rel * 2;

    var color = vec4<f32>(0.0);
    for (var dy = 0u; dy < 4u; dy++) {
        let wy = binomial_weight(dy);
        for (var dx = 0u; dx < 4u; dx++) {
            let wx = binomial_weight(dx);
            let s = sample_input_edge_none(in, src_center + vec2<i32>(i32(dx) - 1, i32(dy) - 1));
            color += s * wx * wy;
        }
    }
    return color;
}

// 2x upscale with phase-aligned [0.75, 0.25] interpolation.
fn upscale_filter(in: FilterVertexOutput) -> vec4<f32> {
    let frag_coord = vec2<u32>(in.position.xy);
    let rel = vec2<i32>(frag_coord - in.dest_offset);

    // Source pixel for this output pixel.
    let src_base = rel / 2;
    let phase = rel % 2; // 0 or 1 in each axis

    // Weights: main pixel gets 0.75, neighbor gets 0.25 per axis.
    // Combined: (0.75*0.75, 0.75*0.25, 0.25*0.75, 0.25*0.25) = (0.5625, 0.1875, 0.1875, 0.0625)
    let dx = vec2<i32>(select(-1i, 1i, phase.x == 1i), 0i);
    let dy = vec2<i32>(0i, select(-1i, 1i, phase.y == 1i));

    let s00 = sample_input_edge_none(in, src_base);
    let s10 = sample_input_edge_none(in, src_base + dx);
    let s01 = sample_input_edge_none(in, src_base + dy);
    let s11 = sample_input_edge_none(in, src_base + dx + dy);

    return s00 * 0.5625 + s10 * 0.1875 + s01 * 0.1875 + s11 * 0.0625;
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

    var max_reach = 0.0;
    if n_linear_taps > 0u {
        // + 1 Since we need one more pixel when doing linear sampling.
        max_reach = offsets[n_linear_taps - 1u] + 1.0;
    }
    let axis_pos = dot(center_in, dir);
    let axis_size = dot(vec2<f32>(in.src_size), dir);
    let all_in_bounds = axis_pos - max_reach >= 0.0
            && axis_pos + max_reach < axis_size;

    // For the best performance, we distinguish between two different paths: In the first path, all pixels
    // we sample are within the bounds of the image, so we can use the GPU-native linear sampling method.
    // However, when blurring border pixels, we can't do that because the image atlas might contain different
    // images there, so we would sample garbage pixels. Therefore, we need to do the interpolation manually there.

    if all_in_bounds {
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
    } else {
        var color = sample_source(in, center_in) * center_weight;
        let fixed = center_in - dir * axis_pos;

        for (var i = 0u; i < n_linear_taps; i++) {
            let w = weights[i];
            let o = offsets[i];
            // positive direction
            let pp = axis_pos + o;
            let pp0 = floor(pp);
            let pt = pp - pp0;
            color += mix(
                sample_source(in, fixed + dir * pp0),
                sample_source(in, fixed + dir * (pp0 + 1.0)),
                pt) * w;
            // negative direction
            let np = axis_pos - o;
            let np0 = floor(np);
            let nt = np - np0;
            color += mix(
                sample_source(in, fixed + dir * np0),
                sample_source(in, fixed + dir * (np0 + 1.0)),
                nt) * w;
        }

        return color;
    }
}

const HORIZONTAL: vec2<f32> = vec2<f32>(1.0, 0.0);
const VERTICAL: vec2<f32> = vec2<f32>(0.0, 1.0);

@fragment
fn fs_main(in: FilterVertexOutput) -> @location(0) vec4<f32> {
    let data = load_filter_data(in.filter_offset);
    let frag_coord = vec2<u32>(in.position.xy);
    let rel_coord = vec2<f32>(frag_coord - in.dest_offset);

    switch in.pass_kind {
        case PASS_COPY: {
            return sample_source(in, rel_coord);
        }
        case PASS_FLOOD: {
            let flood = unpack_flood_filter(data);
            return unpack_color(flood.color);
        }
        case PASS_OFFSET: {
            let filter_type = get_filter_type(data);
            if filter_type == FILTER_TYPE_DROP_SHADOW {
                let shadow = unpack_drop_shadow_filter(data);
                // For drop shadow, use floor to match the old combined offset+blur behavior.
                return sample_source(in, floor(rel_coord - vec2<f32>(shadow.dx, shadow.dy)));
            } else {
                let offset = unpack_offset_filter(data);
                return sample_source(in, round(rel_coord - vec2<f32>(offset.dx, offset.dy)));
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
