// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// The texture that holds the encoded parameters for all filter effects used in the scene.
@group(0) @binding(0) var filter_data: texture_2d<u32>;
// The texture page containing the current ping-pong source for this filter pass.
@group(1) @binding(0) var source_texture: texture_2d<f32>;
// A bilinear sampler.
@group(1) @binding(1) var linear_sampler: sampler;
// Texture containing preserved original, unfiltered layer content for filters
// that need it. Otherwise, this is potentially just a dummy texture.
@group(2) @binding(0) var original_texture: texture_2d<f32>;

// Keep these variables and layouts in sync with the ones in `filter.rs`!

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
const PASS_COLORIZE: u32 = 8u;

// Keep in sync with FILTER_ATLAS_PADDING in vello_hybrid/src/filter.rs.
const FILTER_ATLAS_PADDING: u32 = 6u;

const MAX_TAPS_PER_SIDE: u32 = 3u;

// The layout of the header:
//   bits [0:4]   = filter_type   (5 bits)
//   bits [5:6]   = edge_mode     (2 bits, only for blur filters), currently ignored.
//   bits [7:10]  = n_decimations (4 bits, only for blur filters), only read on the CPU side.
//   bits [11:12] = n_linear_taps (2 bits, only for blur filters)
//   bit  [13]    = composite_original (only for drop shadows), only read on the CPU side.
//   bits [14:32] = reserved for future use

fn load_filter_texel(texel_offset: u32, texel_index: u32) -> vec4<u32> {
    let w = textureDimensions(filter_data).x;
    let flat_index = texel_offset + texel_index;
    return textureLoad(filter_data, vec2(flat_index % w, flat_index / w), 0);
}

/// Filter type stored in the packed header.
fn get_filter_type(texel0: vec4<u32>) -> u32 { return texel0.x & 0x1Fu; }

/// Number of linear taps stored in the packed header for blur filters.
fn get_filter_header_n_linear_taps(texel0: vec4<u32>) -> u32 { return (texel0.x >> 11u) & 0x3u; }

/// Horizontal offset for an offset filter.
fn get_offset_dx(texel0: vec4<u32>) -> f32 { return bitcast<f32>(texel0.y); }

/// Vertical offset for an offset filter.
fn get_offset_dy(texel0: vec4<u32>) -> f32 { return bitcast<f32>(texel0.z); }

/// Flood color packed as RGBA8.
fn get_flood_color(texel0: vec4<u32>) -> u32 { return texel0.y; }

/// Center weight for gaussian blur convolution.
fn get_blur_center_weight(texel0: vec4<u32>) -> f32 { return bitcast<f32>(texel0.y); }

/// Linear sample weights for gaussian blur convolution.
fn get_blur_linear_weights(texel0: vec4<u32>, texel1: vec4<u32>) -> vec3<f32> {
    // Note: This assumes that `MAX_TAPS_PER_SIDE` = 3.
    return vec3<f32>(
        bitcast<f32>(texel0.z),
        bitcast<f32>(texel0.w),
        bitcast<f32>(texel1.x),
    );
}

/// Linear sample offsets for gaussian blur convolution.
fn get_blur_linear_offsets(texel1: vec4<u32>) -> vec3<f32> {
    // Note: This assumes that `MAX_TAPS_PER_SIDE` = 3.
    return vec3<f32>(
        bitcast<f32>(texel1.y),
        bitcast<f32>(texel1.z),
        bitcast<f32>(texel1.w),
    );
}

/// Horizontal offset for a drop shadow filter.
fn get_drop_shadow_dx(texel2: vec4<u32>) -> f32 { return bitcast<f32>(texel2.x); }

/// Vertical offset for a drop shadow filter.
fn get_drop_shadow_dy(texel2: vec4<u32>) -> f32 { return bitcast<f32>(texel2.y); }

/// Drop shadow color packed as RGBA8.
fn get_drop_shadow_color(texel2: vec4<u32>) -> u32 { return texel2.z; }

struct FilterInstanceData {
    @location(0) source_origin: u32,
    @location(1) source_size: u32,
    @location(2) dest_origin: u32,
    @location(3) dest_size: u32,
    @location(4) dest_texture_size: u32,
    @location(5) filter_data_offset: u32,
    @location(6) original_origin: u32,
    @location(7) original_size: u32,
    @location(8) filter_pass_kind: u32,
}

struct FilterVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) filter_data_offset: u32,
    @location(1) @interpolate(flat) source_origin: vec2<u32>,
    @location(2) @interpolate(flat) source_size: vec2<u32>,
    @location(3) @interpolate(flat) dest_origin: vec2<u32>,
    @location(4) @interpolate(flat) dest_size: vec2<u32>,
    @location(5) @interpolate(flat) original_origin: vec2<u32>,
    @location(6) @interpolate(flat) filter_pass_kind: u32,
}

fn unpack_u16_pair(value: u32) -> vec2<u32> {
    return vec2<u32>(value & 0xffffu, value >> 16u);
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    instance: FilterInstanceData
) -> FilterVertexOutput {
    let quad_vertex = vertex_index % 4u;
    let x = f32((quad_vertex & 1u));
    let y = f32((quad_vertex >> 1u));
    let source_origin = unpack_u16_pair(instance.source_origin);
    let source_size = unpack_u16_pair(instance.source_size);
    let dest_origin = unpack_u16_pair(instance.dest_origin);
    let dest_size = unpack_u16_pair(instance.dest_size);
    let dest_texture_size = vec2<f32>(unpack_u16_pair(instance.dest_texture_size));
    let original_origin = unpack_u16_pair(instance.original_origin);
    let original_size = unpack_u16_pair(instance.original_size);

    // Decimated passes only write `dest_size`, but must also clear the bottom/right border
    // of stale pixels to transparent so our bilinear taps don't accidentally sample stale pixels.
    let render_size = min(original_size, dest_size + vec2(FILTER_ATLAS_PADDING));
    let pix_x = f32(dest_origin.x) + x * f32(render_size.x);
    let pix_y = f32(dest_origin.y) + y * f32(render_size.y);

    let ndc_x = pix_x * 2.0 / dest_texture_size.x - 1.0;
    let ndc_y = 1.0 - pix_y * 2.0 / dest_texture_size.y;

    var out: FilterVertexOutput;
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.filter_data_offset = instance.filter_data_offset;
    out.source_origin = source_origin;
    out.source_size = source_size;
    out.dest_origin = dest_origin;
    out.dest_size = dest_size;
    out.original_origin = original_origin;
    out.filter_pass_kind = instance.filter_pass_kind;
    return out;
}

// Sample a pixel from the original texture.
// Note: `rel_cord` needs to be positive and must not exceed the width/height of the image
// that is to be sampled.
fn sample_original(original_origin: vec2<u32>, rel_coord: vec2<f32>) -> vec4<f32> {
    let source_coord = vec2<u32>(vec2<i32>(original_origin) + vec2<i32>(rel_coord));
    return textureLoad(original_texture, source_coord, 0);
}

// Sample a pixel from the source texture.
// Note: `rel_cord` needs to be positive and must not exceed the width/height of the image
// that is to be sampled.
fn sample_source(source_origin: vec2<u32>, rel_coord: vec2<f32>) -> vec4<f32> {
    let source_coord = vec2<u32>(vec2<i32>(source_origin) + vec2<i32>(rel_coord));
    return textureLoad(source_texture, source_coord, 0);
}

// Same as `sample_source`, but with bounds checking.
fn sample_source_checked(
    source_origin: vec2<u32>,
    source_size: vec2<u32>,
    rel_coord: vec2<f32>,
) -> vec4<f32> {
    if rel_coord.x < 0.0 || rel_coord.x >= f32(source_size.x) ||
       rel_coord.y < 0.0 || rel_coord.y >= f32(source_size.y) {
        return vec4<f32>(0.0);
    }

    return sample_source(source_origin, rel_coord);
}

// TODO: Add support for edge modes when blurring. This unfortunately will make it harder/impossible to perform
// filtering with bilinear samplers (since the padding of images is always transparent), hence why this is currently
// not implemented. It's possible, but we should only do it if we really need it.

// Note that `downscale` and `upscale` and `convolve` all use GPU-native bilinear sampling, assuming that there is a large
// enough transparent border around the images (which is currently always the case).

// We need to use `textureSampleLevel` instead of `textureSample` for loops with dynamic
// iteration count so that it works properly in the Direct3D backend.

fn downscale(
    position: vec4<f32>,
    source_origin: vec2<u32>,
    dest_origin: vec2<u32>,
) -> vec4<f32> {
    let frag_coord = vec2<u32>(position.xy);
    let rel = vec2<i32>(frag_coord - dest_origin);
    let source_rel = vec2<f32>(rel * 2);
    let source_texel = vec2<f32>(source_origin) + source_rel;
    let source_texture_size = vec2<f32>(textureDimensions(source_texture));

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

    let s00 = textureSampleLevel(source_texture, linear_sampler, (source_texel + vec2(lo.x, lo.y) + 0.5) / source_texture_size, 0.0);
    let s01 = textureSampleLevel(source_texture, linear_sampler, (source_texel + vec2(lo.x, hi.y) + 0.5) / source_texture_size, 0.0);
    let s10 = textureSampleLevel(source_texture, linear_sampler, (source_texel + vec2(hi.x, lo.y) + 0.5) / source_texture_size, 0.0);
    let s11 = textureSampleLevel(source_texture, linear_sampler, (source_texel + vec2(hi.x, hi.y) + 0.5) / source_texture_size, 0.0);

    // Now we can just average them.
    return (s00 + s01 + s10 + s11) * 0.25;
}

fn upscale(
    position: vec4<f32>,
    source_origin: vec2<u32>,
    dest_origin: vec2<u32>,
) -> vec4<f32> {
    // Same story as for downscaling, but this time even simpler and we can get away with a single texture sample.

    let frag_coord = vec2<u32>(position.xy);
    let rel = vec2<i32>(frag_coord - dest_origin);
    let source_base = vec2<f32>(rel / 2);
    let phase = vec2<f32>(rel % 2);
    let source_texture_size = vec2<f32>(textureDimensions(source_texture));

    // For even phases: 75% of current, 25% of top/left.
    // For odd phases: 75% of current, 25% of bottom/right.
    let sample_offset = select(vec2(-0.25), vec2(0.25), phase == vec2(1.0));
    let source_texel = vec2<f32>(source_origin) + source_base + sample_offset;

    // Yay, just a single sample!
    return textureSampleLevel(source_texture, linear_sampler, (source_texel + 0.5) / source_texture_size, 0.0);
}

fn convolve(
    source_origin: vec2<u32>,
    source_rel: vec2<f32>,
    dir: vec2<f32>,
    n_linear_taps: u32,
    center_weight: f32,
    weights: vec3<f32>,
    offsets: vec3<f32>,
) -> vec4<f32> {
    // See the description in `filter.rs` for a bit more information on how this works. For vello_cpu, we
    // precompute a kernel and then apply separate horizontal/vertical passes to achieve the blurring.
    // For the GPU version, we also do horizontal/vertical filtering in two separate passes, but we precompute
    // new weights as well as a number of fractional offsets, such that we can achieve the same filtering result
    // with much fewer samples, again thanks to bilinear filtering!

    // TODO: Explore whether combining horizontal and vertical filtering is worth it. Likely not worth doing
    // since downscaling/upscaling forms the bottleneck for now.

    let source_texel = vec2<f32>(source_origin) + source_rel;
    let source_texture_size = vec2<f32>(textureDimensions(source_texture));

    // First compute the color contribution of the center pixel.
    var color = textureSampleLevel(source_texture, linear_sampler, (source_texel + 0.5) / source_texture_size, 0.0) * center_weight;

    // See https://github.com/linebender/vello/pull/1601#issuecomment-4323170395 for why we convert
    // into array first. Also see https://github.com/linebender/vello/pull/1605 for why we
    // assign each field separately.

    var weights_arr: array<f32, 3>;
    weights_arr[0] = weights.x;
    weights_arr[1] = weights.y;
    weights_arr[2] = weights.z;

    var offsets_arr: array<f32, 3>;
    offsets_arr[0] = offsets.x;
    offsets_arr[1] = offsets.y;
    offsets_arr[2] = offsets.z;

    // Then, compute and sum the contributions of the adjacent pixels.
    for (var i = 0u; i < n_linear_taps; i++) {
        let w = weights_arr[i];
        let d = dir * offsets_arr[i];
        color += textureSampleLevel(source_texture, linear_sampler, (source_texel + d + 0.5) / source_texture_size, 0.0) * w;
        color += textureSampleLevel(source_texture, linear_sampler, (source_texel - d + 0.5) / source_texture_size, 0.0) * w;
    }

    return color;
}

const HORIZONTAL: vec2<f32> = vec2<f32>(1.0, 0.0);
const VERTICAL: vec2<f32> = vec2<f32>(0.0, 1.0);

@fragment
fn fs_main(
    @location(0) @interpolate(flat) filter_data_offset: u32,
    @location(1) @interpolate(flat) source_origin: vec2<u32>,
    @location(2) @interpolate(flat) source_size: vec2<u32>,
    @location(3) @interpolate(flat) dest_origin: vec2<u32>,
    @location(4) @interpolate(flat) dest_size: vec2<u32>,
    @location(5) @interpolate(flat) original_origin: vec2<u32>,
    @location(6) @interpolate(flat) filter_pass_kind: u32,
    @builtin(position) position: vec4<f32>,
) -> @location(0) vec4<f32> {
    let frag_coord = vec2<u32>(position.xy);
    let rel_coord = vec2<f32>(frag_coord - dest_origin);
    // See the comment in `vs_main`.
    if rel_coord.x >= f32(dest_size.x) || rel_coord.y >= f32(dest_size.y) {
        return vec4<f32>(0.0);
    }

    switch filter_pass_kind {
        case PASS_COPY: {
            return sample_source(source_origin, rel_coord);
        }
        case PASS_FLOOD: {
            let filter_texel0 = load_filter_texel(filter_data_offset, 0u);
            return unpack4x8unorm(get_flood_color(filter_texel0));
        }
        case PASS_OFFSET: {
            let filter_texel0 = load_filter_texel(filter_data_offset, 0u);
            var dxdy: vec2<f32>;

            if get_filter_type(filter_texel0) == FILTER_TYPE_DROP_SHADOW {
                let filter_texel2 = load_filter_texel(filter_data_offset, 2u);
                dxdy = vec2<f32>(get_drop_shadow_dx(filter_texel2), get_drop_shadow_dy(filter_texel2));
            } else {
                dxdy = vec2<f32>(get_offset_dx(filter_texel0), get_offset_dy(filter_texel0));
            }

            // CPU version uses normal round but WGSL round with ties even, so we use floor + 0.5 instead.
            return sample_source_checked(source_origin, source_size, rel_coord - floor(dxdy + 0.5));
        }
        case PASS_DOWNSCALE: {
            return downscale(position, source_origin, dest_origin);
        }
        case PASS_BLUR_H: {
            let filter_texel0 = load_filter_texel(filter_data_offset, 0u);
            let filter_texel1 = load_filter_texel(filter_data_offset, 1u);
            return convolve(
                source_origin,
                rel_coord,
                HORIZONTAL,
                get_filter_header_n_linear_taps(filter_texel0),
                get_blur_center_weight(filter_texel0),
                get_blur_linear_weights(filter_texel0, filter_texel1),
                get_blur_linear_offsets(filter_texel1),
            );
        }
        case PASS_BLUR_V: {
            let filter_texel0 = load_filter_texel(filter_data_offset, 0u);
            let filter_texel1 = load_filter_texel(filter_data_offset, 1u);
            return convolve(
                source_origin,
                rel_coord,
                VERTICAL,
                get_filter_header_n_linear_taps(filter_texel0),
                get_blur_center_weight(filter_texel0),
                get_blur_linear_weights(filter_texel0, filter_texel1),
                get_blur_linear_offsets(filter_texel1),
            );
        }
        case PASS_UPSCALE: {
            return upscale(position, source_origin, dest_origin);
        }
        case PASS_COMPOSITE_DROP_SHADOW: {
            let filter_texel2 = load_filter_texel(filter_data_offset, 2u);
            // Drop shadow composite: colorize blurred result, composite original on top.
            let blurred = sample_source(source_origin, rel_coord);
            let shadow_color = unpack4x8unorm(get_drop_shadow_color(filter_texel2));
            let shadow_result = shadow_color * blurred.a;
            let original = sample_original(original_origin, rel_coord);

            // Simple source-over compositing.
            return original + shadow_result * (1.0 - original.a);
        }
        case PASS_COLORIZE: {
            let filter_texel2 = load_filter_texel(filter_data_offset, 2u);
            let blurred = sample_source(source_origin, rel_coord);
            let shadow_color = unpack4x8unorm(get_drop_shadow_color(filter_texel2));
            return shadow_color * blurred.a;
        }
        // Shouldn't be reached.
        default: {
            return vec4<f32>(0.0);
        }
    }
}
