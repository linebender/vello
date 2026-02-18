// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// The texture that holds the encoded parameters for all filter effects used in the scene.
// Set once per scene, does not change between filter passes.
@group(0) @binding(0) var filter_data: texture_2d<u32>;

// The texture holding the input image we sample from to create the filter effect.
// Changes per filter pass (each layer has its own intermediate texture).
@group(1) @binding(0) var in_tex: texture_2d<f32>;

// Linear sampler for hardware bilinear filtering (used by fast-path convolution).
@group(1) @binding(1) var linear_sampler: sampler;

// The original (unfiltered) content texture, used by drop shadow pass 2 for compositing.
// Only bound during pass 2 (fs_pass_2).
@group(2) @binding(0) var original_tex: texture_2d<f32>;

// Keep these variables and structs in sync with the ones in `filter.rs`!

const FILTER_SIZE_U32: u32 = 12u;
// Since the texture is packed into Uint32.
const TEXELS_PER_FILTER: u32 = FILTER_SIZE_U32 / 4u;

const FILTER_TYPE_OFFSET: u32 = 0u;
const FILTER_TYPE_FLOOD: u32 = 1u;
const FILTER_TYPE_GAUSSIAN_BLUR: u32 = 2u;
const FILTER_TYPE_DROP_SHADOW: u32 = 3u;

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

struct GaussianBlurFilter {
    std_deviation: f32,
    edge_mode: u32,
    n_decimations: u32,
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

fn unpack_gaussian_blur_filter(data: GpuFilterData) -> GaussianBlurFilter {
    let header = data.data[0];
    var weights: array<f32, 3>;
    var offsets: array<f32, 3>;
    for (var i = 0u; i < 3u; i++) {
        weights[i] = bitcast<f32>(data.data[3u + i]);
        offsets[i] = bitcast<f32>(data.data[6u + i]);
    }
    return GaussianBlurFilter(
        bitcast<f32>(data.data[1]),              // std_deviation
        unpack_header_edge_mode(header),         // edge_mode
        unpack_header_n_decimations(header),     // n_decimations
        unpack_header_n_linear_taps(header),     // n_linear_taps
        bitcast<f32>(data.data[2]),              // center_weight
        weights,
        offsets,
    );
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
    // TODO: Is there a more compact way of doing this?
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
    // images there, so we would sample garbage pixels. Therefore, we need to do the interpolational manually there.

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

@fragment
fn fs_pass_1(in: FilterVertexOutput) -> @location(0) vec4<f32> {
    let data = load_filter_data(in.filter_offset);
    let frag_coord = vec2<u32>(in.position.xy);

    // Convert frag_coord from dest atlas space to relative (0..dest_size)
    let rel_coord = vec2<f32>(frag_coord - in.dest_offset);

    let filter_type = get_filter_type(data);

    if filter_type == FILTER_TYPE_FLOOD {
        let flood = unpack_flood_filter(data);
        return apply_flood(flood);
    } else if filter_type == FILTER_TYPE_GAUSSIAN_BLUR {
        let blur = unpack_gaussian_blur_filter(data);
        return gaussian_blur_pass_1(in, rel_coord, blur);
    } else if filter_type == FILTER_TYPE_DROP_SHADOW {
        let shadow = unpack_drop_shadow_filter(data);
        return drop_shadow_pass_1(in, rel_coord, shadow);
    } else {
        let offset = unpack_offset_filter(data);
        return apply_offset(in, rel_coord, offset);
    }
}

@fragment
fn fs_pass_2(in: FilterVertexOutput) -> @location(0) vec4<f32> {
    let data = load_filter_data(in.filter_offset);
    let frag_coord = vec2<u32>(in.position.xy);

    // Convert frag_coord from dest atlas space to relative (0..dest_size)
    let rel_coord = vec2<f32>(frag_coord - in.dest_offset);

    let filter_type = get_filter_type(data);

    if filter_type == FILTER_TYPE_GAUSSIAN_BLUR {
        let blur = unpack_gaussian_blur_filter(data);
        return gaussian_blur_pass_2(in, rel_coord, blur);
    } else if filter_type == FILTER_TYPE_DROP_SHADOW {
        let shadow = unpack_drop_shadow_filter(data);
        return drop_shadow_pass_2(in, rel_coord, shadow);
    }

    // This should never be reached.
    return vec4<f32>(0.0);
}

fn apply_flood(flood: FloodFilter) -> vec4<f32> {
    return unpack_color(flood.color);
}

fn apply_offset(in: FilterVertexOutput, rel_coord: vec2<f32>, offset: OffsetFilter) -> vec4<f32> {
    return sample_source(in, round(rel_coord - vec2<f32>(offset.dx, offset.dy)));
}

const HORIZONTAL: vec2<f32> = vec2<f32>(1.0, 0.0);
const VERTICAL: vec2<f32> = vec2<f32>(0.0, 1.0);

/// Gaussian blur pass 1: horizontal convolution.
fn gaussian_blur_pass_1(in: FilterVertexOutput, rel_coord: vec2<f32>, blur: GaussianBlurFilter) -> vec4<f32> {
    return convolve(in, rel_coord, HORIZONTAL, blur.n_linear_taps, blur.center_weight, blur.linear_weights, blur.linear_offsets);
}

/// Gaussian blur pass 2: vertical convolution.
fn gaussian_blur_pass_2(in: FilterVertexOutput, rel_coord: vec2<f32>, blur: GaussianBlurFilter) -> vec4<f32> {
    return convolve(in, rel_coord, VERTICAL, blur.n_linear_taps, blur.center_weight, blur.linear_weights, blur.linear_offsets);
}

/// Drop shadow pass 1: apply offset then horizontal blur.
/// The offset is rounded to integer to match the CPU path and because the convolution
/// assumes integer-spaced taps.
fn drop_shadow_pass_1(in: FilterVertexOutput, rel_coord: vec2<f32>, shadow: DropShadowFilter) -> vec4<f32> {
    // TODO: Why floor here?
    let offset_center = floor(rel_coord - vec2<f32>(shadow.dx, shadow.dy));
    return convolve(in, offset_center, HORIZONTAL, shadow.n_linear_taps, shadow.center_weight, shadow.linear_weights, shadow.linear_offsets);
}

/// Drop shadow pass 2: vertical blur, colorize, then composite original on top.
fn drop_shadow_pass_2(in: FilterVertexOutput, rel_coord: vec2<f32>, shadow: DropShadowFilter) -> vec4<f32> {
    let blurred = convolve(in, rel_coord, VERTICAL, shadow.n_linear_taps, shadow.center_weight, shadow.linear_weights, shadow.linear_offsets);
    let shadow_color = unpack_color(shadow.color);
    // shadow.color is premultiplied RGBA. Scale it by the blurred alpha to get
    // the final shadow contribution at this pixel.
    let shadow_result = shadow_color * blurred.a;

    // Composite original content over the shadow (premultiplied source-over).
    let original = sample_original(in, rel_coord);
    return original + shadow_result * (1.0 - original.a);
}
