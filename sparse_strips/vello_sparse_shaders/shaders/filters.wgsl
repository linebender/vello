// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// Filter types (must match filter_type module in filter.rs)
const FILTER_TYPE_OFFSET: u32 = 0u;
const FILTER_TYPE_FLOOD: u32 = 1u;
const FILTER_TYPE_GAUSSIAN_BLUR: u32 = 2u;
const FILTER_TYPE_DROP_SHADOW: u32 = 3u;

// Edge modes (must match edge_mode module in filter.rs)
const EDGE_MODE_DUPLICATE: u32 = 0u;
const EDGE_MODE_WRAP: u32 = 1u;
const EDGE_MODE_MIRROR: u32 = 2u;
const EDGE_MODE_NONE: u32 = 3u;

const MAX_KERNEL_SIZE: u32 = 13u;

// Filter data structures - correspond to Gpu* types in filter.rs

struct OffsetFilter {
    dx: f32,
    dy: f32,
}

struct FloodFilter {
    color: u32,
}

struct GaussianBlurFilter {
    std_deviation: f32,
    n_decimations: u32,
    kernel_size: u32,
    edge_mode: u32,
    kernel: array<f32, 13>,
}

struct DropShadowFilter {
    dx: f32,
    dy: f32,
    color: u32,
    edge_mode: u32,
    std_deviation: f32,
    n_decimations: u32,
    kernel_size: u32,
    kernel: array<f32, 13>,
}

// Unpacking functions

fn unpack_offset_filter(filter_data: texture_2d<u32>, base_idx: u32) -> OffsetFilter {
    let texel0 = textureLoad(filter_data, vec2<u32>(base_idx, 0u), 0);
    return OffsetFilter(bitcast<f32>(texel0.x), bitcast<f32>(texel0.y));
}

fn unpack_flood_filter(filter_data: texture_2d<u32>, base_idx: u32) -> FloodFilter {
    let texel0 = textureLoad(filter_data, vec2<u32>(base_idx, 0u), 0);
    return FloodFilter(texel0.x);
}

fn unpack_gaussian_blur_filter(filter_data: texture_2d<u32>, base_idx: u32) -> GaussianBlurFilter {
    let texel0 = textureLoad(filter_data, vec2<u32>(base_idx, 0u), 0);
    let texel1 = textureLoad(filter_data, vec2<u32>(base_idx + 1u, 0u), 0);
    let texel2 = textureLoad(filter_data, vec2<u32>(base_idx + 2u, 0u), 0);
    let texel3 = textureLoad(filter_data, vec2<u32>(base_idx + 3u, 0u), 0);
    let texel4 = textureLoad(filter_data, vec2<u32>(base_idx + 4u, 0u), 0);

    var kernel: array<f32, 13>;
    kernel[0] = bitcast<f32>(texel1.x);
    kernel[1] = bitcast<f32>(texel1.y);
    kernel[2] = bitcast<f32>(texel1.z);
    kernel[3] = bitcast<f32>(texel1.w);
    kernel[4] = bitcast<f32>(texel2.x);
    kernel[5] = bitcast<f32>(texel2.y);
    kernel[6] = bitcast<f32>(texel2.z);
    kernel[7] = bitcast<f32>(texel2.w);
    kernel[8] = bitcast<f32>(texel3.x);
    kernel[9] = bitcast<f32>(texel3.y);
    kernel[10] = bitcast<f32>(texel3.z);
    kernel[11] = bitcast<f32>(texel3.w);
    kernel[12] = bitcast<f32>(texel4.x);

    return GaussianBlurFilter(
        bitcast<f32>(texel0.x),
        texel0.y,
        texel0.z,
        texel0.w,
        kernel
    );
}

fn unpack_drop_shadow_filter(filter_data: texture_2d<u32>, base_idx: u32) -> DropShadowFilter {
    let texel0 = textureLoad(filter_data, vec2<u32>(base_idx, 0u), 0);
    let texel1 = textureLoad(filter_data, vec2<u32>(base_idx + 1u, 0u), 0);
    let texel2 = textureLoad(filter_data, vec2<u32>(base_idx + 2u, 0u), 0);
    let texel3 = textureLoad(filter_data, vec2<u32>(base_idx + 3u, 0u), 0);
    let texel4 = textureLoad(filter_data, vec2<u32>(base_idx + 4u, 0u), 0);
    let texel5 = textureLoad(filter_data, vec2<u32>(base_idx + 5u, 0u), 0);

    var kernel: array<f32, 13>;
    kernel[0] = bitcast<f32>(texel2.x);
    kernel[1] = bitcast<f32>(texel2.y);
    kernel[2] = bitcast<f32>(texel2.z);
    kernel[3] = bitcast<f32>(texel2.w);
    kernel[4] = bitcast<f32>(texel3.x);
    kernel[5] = bitcast<f32>(texel3.y);
    kernel[6] = bitcast<f32>(texel3.z);
    kernel[7] = bitcast<f32>(texel3.w);
    kernel[8] = bitcast<f32>(texel4.x);
    kernel[9] = bitcast<f32>(texel4.y);
    kernel[10] = bitcast<f32>(texel4.z);
    kernel[11] = bitcast<f32>(texel4.w);
    kernel[12] = bitcast<f32>(texel5.x);

    return DropShadowFilter(
        bitcast<f32>(texel0.x),
        bitcast<f32>(texel0.y),
        texel0.z,
        texel0.w,
        bitcast<f32>(texel1.x),
        texel1.y,
        texel1.z,
        kernel
    );
}

// Utility functions

fn unpack_color(packed: u32) -> vec4<f32> {
    return unpack4x8unorm(packed);
}

fn sample_with_edge_mode(
    tex: texture_2d<f32>,
    coord: vec2<i32>,
    tex_size: vec2<i32>,
    edge_mode: u32
) -> vec4<f32> {
    var sample_coord = coord;

    let out_of_bounds = coord.x < 0 || coord.x >= tex_size.x ||
                        coord.y < 0 || coord.y >= tex_size.y;

    if out_of_bounds {
        switch edge_mode {
            case EDGE_MODE_DUPLICATE: {
                sample_coord = clamp(coord, vec2<i32>(0), tex_size - vec2<i32>(1));
            }
            case EDGE_MODE_WRAP: {
                sample_coord = ((coord % tex_size) + tex_size) % tex_size;
            }
            case EDGE_MODE_MIRROR: {
                let period = tex_size * 2;
                var mirrored = ((coord % period) + period) % period;
                if mirrored.x >= tex_size.x {
                    mirrored.x = period.x - 1 - mirrored.x;
                }
                if mirrored.y >= tex_size.y {
                    mirrored.y = period.y - 1 - mirrored.y;
                }
                sample_coord = mirrored;
            }
            case EDGE_MODE_NONE, default: {
                return vec4<f32>(0.0, 0.0, 0.0, 0.0);
            }
        }
    }

    return textureLoad(tex, vec2<u32>(sample_coord), 0);
}

// Filter implementations

fn apply_offset(
    input_tex: texture_2d<f32>,
    frag_coord: vec2<u32>,
    filter: OffsetFilter
) -> vec4<f32> {
    let tex_size = vec2<i32>(textureDimensions(input_tex));
    let src_coord = vec2<i32>(frag_coord) - vec2<i32>(i32(round(filter.dx)), i32(round(filter.dy)));
    return sample_with_edge_mode(input_tex, src_coord, tex_size, EDGE_MODE_NONE);
}

fn apply_flood(filter: FloodFilter) -> vec4<f32> {
    return unpack_color(filter.color);
}

// Shader entry points (placeholders)

struct FilterVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> FilterVertexOutput {
    var out: FilterVertexOutput;
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.tex_coord = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(in: FilterVertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
