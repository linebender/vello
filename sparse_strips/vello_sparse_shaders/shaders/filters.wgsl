// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

const FILTER_TYPE_OFFSET: u32 = 0u;
const FILTER_TYPE_FLOOD: u32 = 1u;
const FILTER_TYPE_GAUSSIAN_BLUR: u32 = 2u;
const FILTER_TYPE_DROP_SHADOW: u32 = 3u;

const EDGE_MODE_DUPLICATE: u32 = 0u;
const EDGE_MODE_WRAP: u32 = 1u;
const EDGE_MODE_MIRROR: u32 = 2u;
const EDGE_MODE_NONE: u32 = 3u;

const MAX_KERNEL_SIZE: u32 = 13u;
const FILTER_SIZE_U32: u32 = 24u;

struct GpuFilterData {
    data: array<u32, 24>,
}

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

fn get_filter_type(filter: GpuFilterData) -> u32 {
    return filter.data[0];
}

fn unpack_offset_filter(filter: GpuFilterData) -> OffsetFilter {
    return OffsetFilter(
        bitcast<f32>(filter.data[1]),
        bitcast<f32>(filter.data[2])
    );
}

fn unpack_flood_filter(filter: GpuFilterData) -> FloodFilter {
    return FloodFilter(filter.data[1]);
}

fn unpack_gaussian_blur_filter(filter: GpuFilterData) -> GaussianBlurFilter {
    var kernel: array<f32, 13>;
    for (var i = 0u; i < 13u; i++) {
        kernel[i] = bitcast<f32>(filter.data[5u + i]);
    }
    return GaussianBlurFilter(
        bitcast<f32>(filter.data[1]),
        filter.data[2],
        filter.data[3],
        filter.data[4],
        kernel
    );
}

fn unpack_drop_shadow_filter(filter: GpuFilterData) -> DropShadowFilter {
    var kernel: array<f32, 13>;
    for (var i = 0u; i < 13u; i++) {
        kernel[i] = bitcast<f32>(filter.data[8u + i]);
    }
    return DropShadowFilter(
        bitcast<f32>(filter.data[1]),
        bitcast<f32>(filter.data[2]),
        filter.data[3],
        filter.data[4],
        bitcast<f32>(filter.data[5]),
        filter.data[6],
        filter.data[7],
        kernel
    );
}

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
