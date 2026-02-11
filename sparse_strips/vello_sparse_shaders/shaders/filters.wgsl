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
const FILTER_TYPE_FLOOD: u32 = 1u;
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

struct FloodFilter {
    color: u32,
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

// Keep in sync with GpuFlood in vello_hybrid/src/filter.rs
fn unpack_flood_filter(data: GpuFilterData) -> FloodFilter {
    return FloodFilter(data.data[1]);
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

fn unpack_color(packed: u32) -> vec4<f32> {
    return unpack4x8unorm(packed);
}

// Keep in sync with FilterInstanceData in vello_hybrid/src/render/wgpu.rs
struct FilterInstanceData {
    @location(0) src_offset: vec2<u32>,
    @location(1) src_size: vec2<u32>,
    @location(2) dest_offset: vec2<u32>,
    @location(3) dest_size: vec2<u32>,
    @location(4) dest_atlas_size: vec2<u32>,
    @location(5) filter_offset: u32,
}

struct FilterVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) filter_offset: u32,
    @location(1) @interpolate(flat) src_offset: vec2<u32>,
    @location(2) @interpolate(flat) src_size: vec2<u32>,
    @location(3) @interpolate(flat) dest_offset: vec2<u32>,
    @location(4) @interpolate(flat) dest_size: vec2<u32>,
    @location(5) @interpolate(flat) dest_atlas_size: vec2<u32>,
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
    return out;
}

@fragment
fn fs_main(in: FilterVertexOutput) -> @location(0) vec4<f32> {
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
        return apply_gaussian_blur_horizontal(in, rel_coord, blur);
    } else {
        let offset = unpack_offset_filter(data);
        return apply_offset(in, rel_coord, offset);
    }
}

fn apply_flood(flood: FloodFilter) -> vec4<f32> {
    return unpack_color(flood.color);
}

fn apply_offset(in: FilterVertexOutput, rel_coord: vec2<f32>, offset: OffsetFilter) -> vec4<f32> {
    // Apply filter offset
    let offset_rel = rel_coord - vec2<f32>(offset.dx, offset.dy);

    // Check bounds in relative space
    if offset_rel.x < 0.0 || offset_rel.x >= f32(in.dest_size.x) ||
       offset_rel.y < 0.0 || offset_rel.y >= f32(in.dest_size.y) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Map to source atlas space
    let src_coord = vec2<i32>(in.src_offset) + vec2<i32>(offset_rel);

    return textureLoad(in_tex, vec2<u32>(src_coord), 0);
}

fn apply_gaussian_blur_horizontal(in: FilterVertexOutput, rel_coord: vec2<f32>, blur: GaussianBlurFilter) -> vec4<f32> {
    let radius = i32(blur.kernel_size / 2u);

    var color = vec4<f32>(0.0);
    for (var i: i32 = -radius; i <= radius; i++) {
        let weight = blur.kernel[i + radius];

        // Sample position in relative space
        let sample_x = rel_coord.x + f32(i);
        let sample_y = rel_coord.y;

        // Check bounds in relative space
        // TODO: Apply edge mode
        if sample_x >= 0.0 && sample_x < f32(in.dest_size.x) &&
           sample_y >= 0.0 && sample_y < f32(in.dest_size.y) {
            // Map to source atlas space
            let src_coord = vec2<i32>(in.src_offset) + vec2<i32>(i32(sample_x), i32(sample_y));
            color += textureLoad(in_tex, vec2<u32>(src_coord), 0) * weight;
        }
    }

    return color;
}

fn apply_gaussian_blur_vertical(in: FilterVertexOutput, rel_coord: vec2<f32>, blur: GaussianBlurFilter) -> vec4<f32> {
    let radius = i32(blur.kernel_size / 2u);

    var color = vec4<f32>(0.0);
    for (var i: i32 = -radius; i <= radius; i++) {
        let weight = blur.kernel[i + radius];

        // Sample position in relative space
        let sample_x = rel_coord.x;
        let sample_y = rel_coord.y + f32(i);

        // Check bounds in relative space
        // TODO: Apply edge mode
        if sample_x >= 0.0 && sample_x < f32(in.dest_size.x) &&
           sample_y >= 0.0 && sample_y < f32(in.dest_size.y) {
            // Map to source atlas space
            let src_coord = vec2<i32>(in.src_offset) + vec2<i32>(i32(sample_x), i32(sample_y));
            color += textureLoad(in_tex, vec2<u32>(src_coord), 0) * weight;
        }
    }

    return color;
}

@fragment
fn fs_main_vertical(in: FilterVertexOutput) -> @location(0) vec4<f32> {
    let data = load_filter_data(in.filter_offset);
    let frag_coord = vec2<u32>(in.position.xy);

    // Convert frag_coord from dest atlas space to relative (0..dest_size)
    let rel_coord = vec2<f32>(frag_coord - in.dest_offset);

    let filter_type = get_filter_type(data);

    if filter_type == FILTER_TYPE_GAUSSIAN_BLUR {
        let blur = unpack_gaussian_blur_filter(data);
        return apply_gaussian_blur_vertical(in, rel_coord, blur);
    }

    // This should never be reached.
    return vec4<f32>(0.0);
}

// --- FOR LATER ---

// const FILTER_TYPE_DROP_SHADOW: u32 = 3u;

// // Keep in sync with EdgeMode in vello_common/src/filter_effects.rs
// // and edge_mode module in vello_hybrid/src/filter.rs
// const EDGE_MODE_DUPLICATE: u32 = 0u;
// const EDGE_MODE_WRAP: u32 = 1u;
// const EDGE_MODE_MIRROR: u32 = 2u;
// const EDGE_MODE_NONE: u32 = 3u;

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
