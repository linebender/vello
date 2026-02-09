// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// The texture that holds the encoded parameters for all filter effects used in the scene.
// Set once per scene, does not change between filter passes.
@group(0) @binding(0) var filter_data: texture_2d<u32>;

// The texture holding the input image we sample from to create the filter effect.
// Changes per filter pass (each layer has its own intermediate texture).
@group(1) @binding(0) var in_tex: texture_2d<f32>;

// Keep in sync with FILTER_SIZE_U32 in vello_hybrid/src/filter.rs
const FILTER_SIZE_U32: u32 = 24u;
// Since the texture is packed into Uint32.
const TEXELS_PER_FILTER: u32 = FILTER_SIZE_U32 / 4u;

// Keep in sync with filter_type module in vello_hybrid/src/filter.rs
const FILTER_TYPE_OFFSET: u32 = 0u;
const FILTER_TYPE_GAUSSIAN_BLUR: u32 = 2u;

// Keep in sync with MAX_KERNEL_SIZE in vello_common/src/filter/gaussian_blur.rs
const MAX_KERNEL_SIZE: u32 = 13u;

struct GpuFilterData {
    data: array<u32, 24>,
}

struct OffsetFilter {
    dx: f32,
    dy: f32,
}

// Keep in sync with GpuGaussianBlur in vello_hybrid/src/filter.rs
struct GaussianBlurFilter {
    std_deviation: f32,
    n_decimations: u32,
    kernel_size: u32,
    edge_mode: u32,
    kernel: array<f32, 13>,
}

fn get_filter_type(data: GpuFilterData) -> u32 {
    return data.data[0];
}

fn unpack_offset_filter(data: GpuFilterData) -> OffsetFilter {
    return OffsetFilter(
        bitcast<f32>(data.data[1]),
        bitcast<f32>(data.data[2])
    );
}

// Keep in sync with GpuGaussianBlur in vello_hybrid/src/filter.rs
fn unpack_gaussian_blur_filter(data: GpuFilterData) -> GaussianBlurFilter {
    var kernel: array<f32, 13>;
    for (var i = 0u; i < 13u; i++) {
        kernel[i] = bitcast<f32>(data.data[5u + i]);
    }
    return GaussianBlurFilter(
        bitcast<f32>(data.data[1]),
        data.data[2],
        data.data[3],
        data.data[4],
        kernel
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

struct FilterVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) filter_offset: u32,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> FilterVertexOutput {
    // The filter index is encoded in the vertex range: draw(n*4..(n+1)*4, 0..1).
    let filter_index = vertex_index / 4u;
    let quad_vertex = vertex_index % 4u;
    let x = f32((quad_vertex << 1u) & 2u);
    let y = f32(quad_vertex & 2u);
    var out: FilterVertexOutput;
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.filter_offset = filter_index * TEXELS_PER_FILTER;
    return out;
}

@fragment
fn fs_main(in: FilterVertexOutput) -> @location(0) vec4<f32> {
    let data = load_filter_data(in.filter_offset);
    let frag_coord = vec2<u32>(in.position.xy);
    let filter_type = get_filter_type(data);

    if filter_type == FILTER_TYPE_GAUSSIAN_BLUR {
        let blur = unpack_gaussian_blur_filter(data);
        return apply_gaussian_blur_horizontal(frag_coord, blur);
    } else {
        let offset = unpack_offset_filter(data);
        return apply_offset(frag_coord, offset);
    }
}

fn apply_offset(frag_coord: vec2<u32>, offset: OffsetFilter) -> vec4<f32> {
    let tex_size = vec2<i32>(textureDimensions(in_tex));
    // TODO: Do offset rounding on the CPU? Should save work on the GPU.
    let src_coord = vec2<i32>(frag_coord) - vec2<i32>(i32(round(offset.dx)), i32(round(offset.dy)));

    if src_coord.x < 0 || src_coord.x >= tex_size.x || src_coord.y < 0 || src_coord.y >= tex_size.y {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    return textureLoad(in_tex, vec2<u32>(src_coord), 0);
}

fn apply_gaussian_blur_horizontal(frag_coord: vec2<u32>, blur: GaussianBlurFilter) -> vec4<f32> {
    let tex_size = vec2<i32>(textureDimensions(in_tex));
    let radius = i32(blur.kernel_size / 2u);

    var color = vec4<f32>(0.0);
    for (var i: i32 = -radius; i <= radius; i++) {
        let weight = blur.kernel[i + radius];
        let src_x = i32(frag_coord.x) + i;
        let src_y = i32(frag_coord.y);

        // TODO: Apply edge mode
        if src_x >= 0 && src_x < tex_size.x && src_y >= 0 && src_y < tex_size.y {
            color += textureLoad(in_tex, vec2<u32>(vec2<i32>(src_x, src_y)), 0) * weight;
        }
    }

    return color;
}

fn apply_gaussian_blur_vertical(frag_coord: vec2<u32>, blur: GaussianBlurFilter) -> vec4<f32> {
    let tex_size = vec2<i32>(textureDimensions(in_tex));
    let radius = i32(blur.kernel_size / 2u);

    var color = vec4<f32>(0.0);
    for (var i: i32 = -radius; i <= radius; i++) {
        let weight = blur.kernel[i + radius];
        let src_x = i32(frag_coord.x);
        let src_y = i32(frag_coord.y) + i;

        // TODO: Apply edge mode
        if src_x >= 0 && src_x < tex_size.x && src_y >= 0 && src_y < tex_size.y {
            color += textureLoad(in_tex, vec2<u32>(vec2<i32>(src_x, src_y)), 0) * weight;
        }
    }

    return color;
}

@fragment
fn fs_main_vertical(in: FilterVertexOutput) -> @location(0) vec4<f32> {
    let data = load_filter_data(in.filter_offset);
    let frag_coord = vec2<u32>(in.position.xy);
    let filter_type = get_filter_type(data);

    if filter_type == FILTER_TYPE_GAUSSIAN_BLUR {
        let blur = unpack_gaussian_blur_filter(data);
        return apply_gaussian_blur_vertical(frag_coord, blur);
    }

    // This should never be reached.
    return vec4<f32>(0.0);
}

// --- FOR LATER ---

// const FILTER_TYPE_FLOOD: u32 = 1u;
// const FILTER_TYPE_DROP_SHADOW: u32 = 3u;

// // Keep in sync with EdgeMode in vello_common/src/filter_effects.rs
// // and edge_mode module in vello_hybrid/src/filter.rs
// const EDGE_MODE_DUPLICATE: u32 = 0u;
// const EDGE_MODE_WRAP: u32 = 1u;
// const EDGE_MODE_MIRROR: u32 = 2u;
// const EDGE_MODE_NONE: u32 = 3u;

// struct FloodFilter {
//     color: u32,
// }

// struct DropShadowFilter {
//     dx: f32,
//     dy: f32,
//     color: u32,
//     edge_mode: u32,
//     std_deviation: f32,
//     n_decimations: u32,
//     kernel_size: u32,
//     kernel: array<f32, 13>,
// }

// // Keep in sync with GpuFlood in vello_hybrid/src/filter.rs
// fn unpack_flood_filter(data: GpuFilterData) -> FloodFilter {
//     return FloodFilter(data.data[1]);
// }

// // Keep in sync with GpuDropShadow in vello_hybrid/src/filter.rs
// fn unpack_drop_shadow_filter(data: GpuFilterData) -> DropShadowFilter {
//     var kernel: array<f32, 13>;
//     for (var i = 0u; i < 13u; i++) {
//         kernel[i] = bitcast<f32>(data.data[8u + i]);
//     }
//     return DropShadowFilter(
//         bitcast<f32>(data.data[1]),
//         bitcast<f32>(data.data[2]),
//         data.data[3],
//         data.data[4],
//         bitcast<f32>(data.data[5]),
//         data.data[6],
//         data.data[7],
//         kernel
//     );
// }

// fn unpack_color(packed: u32) -> vec4<f32> {
//     return unpack4x8unorm(packed);
// }

// fn sample_with_edge_mode(
//     tex: texture_2d<f32>,
//     coord: vec2<i32>,
//     tex_size: vec2<i32>,
//     edge_mode: u32
// ) -> vec4<f32> {
//     var sample_coord = coord;

//     let out_of_bounds = coord.x < 0 || coord.x >= tex_size.x ||
//                         coord.y < 0 || coord.y >= tex_size.y;

//     if out_of_bounds {
//         switch edge_mode {
//             case EDGE_MODE_DUPLICATE: {
//                 sample_coord = clamp(coord, vec2<i32>(0), tex_size - vec2<i32>(1));
//             }
//             case EDGE_MODE_WRAP: {
//                 sample_coord = ((coord % tex_size) + tex_size) % tex_size;
//             }
//             case EDGE_MODE_MIRROR: {
//                 let period = tex_size * 2;
//                 var mirrored = ((coord % period) + period) % period;
//                 if mirrored.x >= tex_size.x {
//                     mirrored.x = period.x - 1 - mirrored.x;
//                 }
//                 if mirrored.y >= tex_size.y {
//                     mirrored.y = period.y - 1 - mirrored.y;
//                 }
//                 sample_coord = mirrored;
//             }
//             case EDGE_MODE_NONE, default: {
//                 return vec4<f32>(0.0, 0.0, 0.0, 0.0);
//             }
//         }
//     }

//     return textureLoad(tex, vec2<u32>(sample_coord), 0);
// }

// fn apply_flood(flood: FloodFilter) -> vec4<f32> {
//     return unpack_color(flood.color);
// }
