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
const PASS_COMPOSITE_DROP_SHADOW: u32 = 7u;

const MAX_TAPS_PER_SIDE: u32 = 3u;

// A type erased instance of a filter containing the values of all parameters.
struct GpuFilterData {
    data: array<u32, FILTER_SIZE_U32>
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
    let quad_vertex = vertex_index % 4u;
    let x = f32((quad_vertex & 1u));
    let y = f32((quad_vertex >> 1u));

    // Note: We are using `original_size` instead of `dest_size` on purpose here. When allocating the regions
    // in the atlas, we always allocate the same size as is used by the original texture. However, `dest_size`
    // can be smaller than `original_size`, for example because we applied a decimation pass for gaussian blurs.
    // However, in the vertex shader we ALWAYS cover the whole region instead of just the destination size.
    // The reason is that we need to make sure that all unaffected pixels are set to transparent, which is important
    // because some filters assume that the border pixels are transparent. In the fragment shader, we have a shortcut
    // to check whether the pixel lies outside of the destination region, in which case we just return a transparent
    // pixel instead of doing actual computational work.
    let pix_x = f32(instance.dest_offset.x) + x * f32(instance.original_size.x);
    let pix_y = f32(instance.dest_offset.y) + y * f32(instance.original_size.y);

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

// Sample a pixel from the original texture.
// Note: `rel_cord` needs to be positive and must not exceed the width/height of the image
// that is to be sampled.
fn sample_original(in: FilterVertexOutput, rel_coord: vec2<f32>) -> vec4<f32> {
    let src_coord = vec2<u32>(vec2<i32>(in.original_offset) + vec2<i32>(rel_coord));
    return textureLoad(original_tex, src_coord, 0);
}

// Sample a pixel from the input texture.
// Note: `rel_cord` needs to be positive and must not exceed the width/height of the image
// that is to be sampled.
fn sample_input(in: FilterVertexOutput, rel_coord: vec2<f32>) -> vec4<f32> {
    let src_coord = vec2<u32>(vec2<i32>(in.src_offset) + vec2<i32>(rel_coord));
    return textureLoad(in_tex, src_coord, 0);
}

// Same as `sample_input`, but with bounds checking.
fn sample_input_checked(in: FilterVertexOutput, rel_coord: vec2<f32>) -> vec4<f32> {
    if rel_coord.x < 0.0 || rel_coord.x >= f32(in.src_size.x) ||
       rel_coord.y < 0.0 || rel_coord.y >= f32(in.src_size.y) {
        return vec4<f32>(0.0);
    }

    return sample_input(in, rel_coord);
}

// TODO: Add support for edge modes when blurring. This unfortunately will make it harder/impossible to perform
// filtering with bilinear samplers (since the padding of images is always transparent), hence why this is currently
// not implemented. It's possible, but we should only do it if we really need it.

// Note that `downscale` and `upscale` and `convolve` all use GPU-native bilinear sampling, assuming that there is a large
// enough transparent border around the images (which is currently always the case).

// We need to use `textureSampleLevel` instead of `textureSample` for loops with dynamic
// iteration count so that it works properly in the Direct3D backend.

fn downscale(in: FilterVertexOutput) -> vec4<f32> {
    let frag_coord = vec2<u32>(in.position.xy);
    let rel = vec2<i32>(frag_coord - in.dest_offset);
    let src_rel = vec2<f32>(rel * 2);
    let src_texel = vec2<f32>(in.src_offset) + src_rel;
    let tex_size = vec2<f32>(textureDimensions(in_tex));

    // Overall, this follows the same approach that is used by the CPU, where a [1,3,3,1]/8 filter
    // is applied in each direction to downscale. However, on the GPU we have native bilinear sampling, which
    // makes things a lot easier for us! Instead of splitting into vertical/horizontal passes we can combine them
    // both into a single pass and determine the sampling points in such a way that the number of total samples
    // is reduced from 16 to just 4.

    // Our sample points are like this [src - 1, src, src + 1, src + 2]. Therefore, to achieve the weighting
    // [1,3,3,1], the left sample points needs to be shifted 0.25 to the left, and
    // the right sample point 1.25 to the right.
    let lo = vec2<f32>(-0.25);
    let hi = vec2<f32>( 1.25);

    let s00 = textureSampleLevel(in_tex, linear_sampler, (src_texel + vec2(lo.x, lo.y) + 0.5) / tex_size, 0.0);
    let s01 = textureSampleLevel(in_tex, linear_sampler, (src_texel + vec2(lo.x, hi.y) + 0.5) / tex_size, 0.0);
    let s10 = textureSampleLevel(in_tex, linear_sampler, (src_texel + vec2(hi.x, lo.y) + 0.5) / tex_size, 0.0);
    let s11 = textureSampleLevel(in_tex, linear_sampler, (src_texel + vec2(hi.x, hi.y) + 0.5) / tex_size, 0.0);

    // Now we can just average them.
    return (s00 + s01 + s10 + s11) * 0.25;
}

fn upscale(in: FilterVertexOutput) -> vec4<f32> {
    // Same story as for downscaling, but this time even simpler and we can get away with a single texture sample.

    let frag_coord = vec2<u32>(in.position.xy);
    let rel = vec2<i32>(frag_coord - in.dest_offset);
    let src_base = vec2<f32>(rel / 2);
    let phase = vec2<f32>(rel % 2);
    let tex_size = vec2<f32>(textureDimensions(in_tex));

    // For even phases: 75% of current, 25% of top/left.
    // For odd phases: 75% of current, 25% of bottom/right.
    let sample_offset = select(vec2(-0.25), vec2(0.25), phase == vec2(1.0));
    let src_texel = vec2<f32>(in.src_offset) + src_base + sample_offset;

    // Yay, just a single sample!
    return textureSampleLevel(in_tex, linear_sampler, (src_texel + 0.5) / tex_size, 0.0);
}

fn convolve(
    in: FilterVertexOutput,
    src_rel: vec2<f32>,
    dir: vec2<f32>,
    n_linear_taps: u32,
    center_weight: f32,
    weights: array<f32, 3>,
    offsets: array<f32, 3>,
) -> vec4<f32> {
    // See the description in `filter.rs` for a bit more information on how this works. For vello_cpu, we
    // precompute a kernel and then apply separate horizontal/vertical passes to achieve the blurring.
    // For the GPU version, we also do horizontal/vertical filtering in two separate passes, but we precompute
    // new weights as well as a number of fractional offsets, such that we can achieve the same filtering result
    // with much fewer samples, again thanks to bilinear filtering!

    // TODO: Explore whether combining horizontal and vertical filtering is worth it. Likely not worth doing
    // since downscaling/upscaling forms the bottleneck for now.

    let src_texel = vec2<f32>(in.src_offset) + src_rel;
    let tex_size = vec2<f32>(textureDimensions(in_tex));

    // First compute the color contribution of the center pixel.
    var color = textureSampleLevel(in_tex, linear_sampler, (src_texel + 0.5) / tex_size, 0.0) * center_weight;

    // Then, compute and sum the contributions of the adjacent pixels.
    for (var i = 0u; i < n_linear_taps; i++) {
        let w = weights[i];
        let d = dir * offsets[i];
        color += textureSampleLevel(in_tex, linear_sampler, (src_texel + d + 0.5) / tex_size, 0.0) * w;
        color += textureSampleLevel(in_tex, linear_sampler, (src_texel - d + 0.5) / tex_size, 0.0) * w;
    }

    return color;
}

const HORIZONTAL: vec2<f32> = vec2<f32>(1.0, 0.0);
const VERTICAL: vec2<f32> = vec2<f32>(0.0, 1.0);

@fragment
fn fs_main(in: FilterVertexOutput) -> @location(0) vec4<f32> {
    let frag_coord = vec2<u32>(in.position.xy);
    let rel_coord = vec2<f32>(frag_coord - in.dest_offset);

    // See the comment in `vs_main`.
    if rel_coord.x >= f32(in.dest_size.x) || rel_coord.y >= f32(in.dest_size.y) {
        return vec4<f32>(0.0);
    }

    switch in.pass_kind {
        case PASS_COPY: {
            return sample_input(in, rel_coord);
        }
        case PASS_FLOOD: {
            let data = load_filter_data(in.filter_offset);
            let flood = unpack_flood_filter(data);

            return unpack4x8unorm(flood.color);
        }
        case PASS_OFFSET: {
            let data = load_filter_data(in.filter_offset);
            let filter_type = unpack_filter_type(data);
            var dxdy: vec2<f32>;

            if filter_type == FILTER_TYPE_DROP_SHADOW {
                let shadow = unpack_drop_shadow_filter(data);
                dxdy = vec2<f32>(shadow.dx, shadow.dy);
            } else {
                let offset = unpack_offset_filter(data);
                dxdy = vec2<f32>(offset.dx, offset.dy);
            }

            // CPU version uses normal round but WGSL round with ties even, so we use floor + 0.5 instead.
            return sample_input_checked(in, rel_coord - floor(dxdy + 0.5));
        }
        case PASS_DOWNSCALE: {
            return downscale(in);
        }
        case PASS_BLUR_H: {
            let data = load_filter_data(in.filter_offset);
            let blur = unpack_blur_params(data);
            return convolve(in, rel_coord, HORIZONTAL, blur.n_linear_taps, blur.center_weight, blur.linear_weights, blur.linear_offsets);
        }
        case PASS_BLUR_V: {
            let data = load_filter_data(in.filter_offset);
            let blur = unpack_blur_params(data);
            return convolve(in, rel_coord, VERTICAL, blur.n_linear_taps, blur.center_weight, blur.linear_weights, blur.linear_offsets);
        }
        case PASS_UPSCALE: {
            return upscale(in);
        }
        case PASS_COMPOSITE_DROP_SHADOW: {
            let data = load_filter_data(in.filter_offset);
            // Drop shadow composite: colorize blurred result, composite original on top.
            let shadow = unpack_drop_shadow_filter(data);
            let blurred = sample_input(in, rel_coord);
            let shadow_color = unpack4x8unorm(shadow.color);
            let shadow_result = shadow_color * blurred.a;
            let original = sample_original(in, rel_coord);

            // Simple source-over compositing.
            return original + shadow_result * (1.0 - original.a);
        }
        // Shouldn't be reached.
        default: {
            return vec4<f32>(0.0);
        }
    }
}
