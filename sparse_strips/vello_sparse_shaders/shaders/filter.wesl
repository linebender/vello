// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

import package::filters::blur::{BLUR_HORIZONTAL, BLUR_VERTICAL, gaussian_blur};
import package::filters::drop_shadow::{
    colorize_drop_shadow,
    composite_drop_shadow,
    get_drop_shadow_offset
};
import package::filters::scale::{downscale, upscale};
import package::helpers::geometry::{pixel_to_ndc, quad_corner};
import package::helpers::texture::flat_index_to_texture_coord;
import package::helpers::util::unpack_u16_pair;

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

// Filters are packed with variable stride: each filter occupies only as many
// 16-byte texels as its parameters need, addressed via `filter_data_offset`.

const FILTER_TYPE_OFFSET: u32 = 0u;
const FILTER_TYPE_FLOOD: u32 = 1u;
const FILTER_TYPE_GAUSSIAN_BLUR: u32 = 2u;
const FILTER_TYPE_DROP_SHADOW: u32 = 3u;
const FILTER_TYPE_COLOR_MATRIX: u32 = 4u;

const PASS_COPY: u32 = 0u;
const PASS_FLOOD: u32 = 1u;
const PASS_OFFSET: u32 = 2u;
const PASS_DOWNSCALE: u32 = 3u;
const PASS_BLUR_H: u32 = 4u;
const PASS_BLUR_V: u32 = 5u;
const PASS_UPSCALE: u32 = 6u;
const PASS_COMPOSITE_DROP_SHADOW: u32 = 7u;
const PASS_COLORIZE: u32 = 8u;
const PASS_COLOR_MATRIX: u32 = 9u;

// Keep in sync with FILTER_ATLAS_PADDING in vello_hybrid/src/filter.rs.
const FILTER_ATLAS_PADDING: u32 = 6u;

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
    return textureLoad(filter_data, flat_index_to_texture_coord(flat_index, w), 0);
}

/// Filter type stored in the packed header.
fn get_filter_type(texel0: vec4<u32>) -> u32 { return texel0.x & 0x1Fu; }

/// Horizontal offset for an offset filter.
fn get_offset_dx(texel0: vec4<u32>) -> f32 { return bitcast<f32>(texel0.y); }

/// Vertical offset for an offset filter.
fn get_offset_dy(texel0: vec4<u32>) -> f32 { return bitcast<f32>(texel0.z); }

/// Flood color packed as RGBA8.
fn get_flood_color(texel0: vec4<u32>) -> u32 { return texel0.y; }

struct FilterInstanceData {
    @location(0)
    source_origin: u32,
    @location(1)
    source_size: u32,
    @location(2)
    dest_origin: u32,
    @location(3)
    dest_size: u32,
    @location(4)
    dest_texture_size: u32,
    @location(5)
    filter_data_offset: u32,
    @location(6)
    original_origin: u32,
    @location(7)
    original_size: u32,
    @location(8)
    filter_pass_kind: u32,
}

struct FilterVertexOutput {
    @builtin(position)
    position: vec4<f32>,
    @location(0) @interpolate(flat)
    filter_data_offset: u32,
    @location(1) @interpolate(flat)
    source_origin: vec2<u32>,
    @location(2) @interpolate(flat)
    source_size: vec2<u32>,
    @location(3) @interpolate(flat)
    dest_origin: vec2<u32>,
    @location(4) @interpolate(flat)
    dest_size: vec2<u32>,
    @location(5) @interpolate(flat)
    original_origin: vec2<u32>,
    @location(6) @interpolate(flat)
    filter_pass_kind: u32,
}

@vertex
fn vs_main(
    @builtin(vertex_index)vertex_index: u32,
    instance: FilterInstanceData
) -> FilterVertexOutput {
    let corner = quad_corner(vertex_index);
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
    let pixel = vec2<f32>(dest_origin) + corner * vec2<f32>(render_size);
    let ndc = pixel_to_ndc(pixel, dest_texture_size);

    var out: FilterVertexOutput;
    out.position = vec4<f32>(ndc, 0.0, 1.0);
    out.filter_data_offset = instance.filter_data_offset;
    out.source_origin = source_origin;
    out.source_size = source_size;
    out.dest_origin = dest_origin;
    out.dest_size = dest_size;
    out.original_origin = original_origin;
    out.filter_pass_kind = instance.filter_pass_kind;
    return out;
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

// One output channel of the 4x5 color matrix: the five packed coefficients
// (RGBA weights + constant offset) applied to a straight-alpha color, clamped.
fn color_matrix_row(m0: u32, m1: u32, m2: u32, m3: u32, m4: u32, color: vec4<f32>) -> f32 {
    return clamp(
        bitcast<f32>(m0) * color.r +
        bitcast<f32>(m1) * color.g +
        bitcast<f32>(m2) * color.b +
        bitcast<f32>(m3) * color.a +
        bitcast<f32>(m4),
        0.0,
        1.0,
    );
}

// The `matrix[0..20]` values are packed after the `header` u32, so `matrix[i]`
// lives at u32 index `i + 1`: texels 0-5 hold header, matrix[0..20], padding.
fn apply_color_matrix(filter_data_offset: u32, pixel: vec4<f32>) -> vec4<f32> {
    let t0 = load_filter_texel(filter_data_offset, 0u);
    let t1 = load_filter_texel(filter_data_offset, 1u);
    let t2 = load_filter_texel(filter_data_offset, 2u);
    let t3 = load_filter_texel(filter_data_offset, 3u);
    let t4 = load_filter_texel(filter_data_offset, 4u);
    let t5 = load_filter_texel(filter_data_offset, 5u);

    var rgb = vec3<f32>(0.0);
    if pixel.a > 0.0 {
        rgb = pixel.rgb / pixel.a;
    }
    let c = vec4<f32>(rgb, pixel.a);

    let out_r = color_matrix_row(t0.y, t0.z, t0.w, t1.x, t1.y, c);
    let out_g = color_matrix_row(t1.z, t1.w, t2.x, t2.y, t2.z, c);
    let out_b = color_matrix_row(t2.w, t3.x, t3.y, t3.z, t3.w, c);
    let out_a = color_matrix_row(t4.x, t4.y, t4.z, t4.w, t5.x, c);

    return vec4<f32>(vec3<f32>(out_r, out_g, out_b) * out_a, out_a);
}

@fragment
fn fs_main(
    @location(0) @interpolate(flat)filter_data_offset: u32,
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
                dxdy = get_drop_shadow_offset(filter_texel2);
            } else {
                dxdy = vec2<f32>(get_offset_dx(filter_texel0), get_offset_dy(filter_texel0));
            }

            // CPU version uses normal round but WGSL round with ties even, so we use floor + 0.5 instead.
            return sample_source_checked(source_origin, source_size, rel_coord - floor(dxdy + 0.5));
        }
        case PASS_DOWNSCALE: {
            return downscale(
                source_texture,
                linear_sampler,
                position,
                source_origin,
                dest_origin,
            );
        }
        case PASS_BLUR_H: {
            let filter_texel0 = load_filter_texel(filter_data_offset, 0u);
            let filter_texel1 = load_filter_texel(filter_data_offset, 1u);
            return gaussian_blur(
                source_texture,
                linear_sampler,
                source_origin,
                rel_coord,
                BLUR_HORIZONTAL,
                filter_texel0,
                filter_texel1,
            );
        }
        case PASS_BLUR_V: {
            let filter_texel0 = load_filter_texel(filter_data_offset, 0u);
            let filter_texel1 = load_filter_texel(filter_data_offset, 1u);
            return gaussian_blur(
                source_texture,
                linear_sampler,
                source_origin,
                rel_coord,
                BLUR_VERTICAL,
                filter_texel0,
                filter_texel1,
            );
        }
        case PASS_UPSCALE: {
            return upscale(
                source_texture,
                linear_sampler,
                position,
                source_origin,
                dest_origin,
            );
        }
        case PASS_COMPOSITE_DROP_SHADOW: {
            let filter_texel2 = load_filter_texel(filter_data_offset, 2u);
            return composite_drop_shadow(
                source_texture,
                original_texture,
                source_origin,
                original_origin,
                rel_coord,
                filter_texel2,
            );
        }
        case PASS_COLORIZE: {
            let filter_texel2 = load_filter_texel(filter_data_offset, 2u);
            return colorize_drop_shadow(
                source_texture,
                source_origin,
                rel_coord,
                filter_texel2,
            );
        }
        case PASS_COLOR_MATRIX: {
            return apply_color_matrix(filter_data_offset, sample_source(source_origin, rel_coord));
        }
        // Shouldn't be reached.
        default: {
            return vec4<f32>(0.0);
        }
    }
}
