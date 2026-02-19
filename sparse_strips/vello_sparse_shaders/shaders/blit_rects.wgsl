// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// Instanced texture blitting shader with affine transform support.
//
// Each `BlitInstance` represents a (maybe rotated) image rectangle that copies
// a region from the image atlas directly to the screen. This shader does not
// support blending and the like and is used as a fast path for blitting from
// textures.
//
// As is described in [`GpuBlitRect`] and [`resolve_blit_rect`], the shader uses
// 2 vectors to define the rect-to-screen transform as defined by:
//
// ```
//   vertex(x, y) = center + col0 * (x - 0.5) + col1 * (y - 0.5)
// ```
//
// For more information, see [`GpuBlitRect`] and [`resolve_blit_rect`].

struct BlitConfig {
    /// Viewport width in pixels.
    width: u32,
    /// Viewport height in pixels.
    height: u32,
    /// Padding for 16-byte alignment (required by WebGL2).
    _padding0: u32,
    _padding1: u32,
}

struct BlitInstance {
    /// See [`GpuBlitRect`] and [`resolve_blit_rect`].
    @location(0) col0: vec2<f32>,
    /// See [`GpuBlitRect`] and [`resolve_blit_rect`].
    @location(1) col1: vec2<f32>,
    /// Screen-space center of the quad: [x, y]
    @location(2) center: vec2<f32>,
    /// Packed atlas source offset: u | (v << 16)
    @location(3) src_xy: u32,
    /// Packed atlas source size: w | (h << 16)
    @location(4) src_wh: u32,
    /// Atlas layer index
    @location(5) atlas_index: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    /// UV coordinates for atlas sampling (in texels).
    @location(0) uv: vec2<f32>,
    /// Atlas layer index (passed flat to fragment).
    @location(1) @interpolate(flat) atlas_index: u32,
}

@group(0) @binding(0) var<uniform> config: BlitConfig;
@group(0) @binding(1) var atlas_texture_array: texture_2d_array<f32>;
@group(0) @binding(2) var atlas_sampler: sampler;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    instance: BlitInstance,
) -> VertexOutput {
    // Map vertex_index (0-3) to quad corners:
    // 0 → (0,0), 1 → (1,0), 2 → (0,1), 3 → (1,1)
    let x = f32(vertex_index & 1u);
    let y = f32(vertex_index >> 1u);

    // Screen pixel position using `col[0|1]` vectors.
    let pixel = instance.center
        + instance.col0 * (x - 0.5)
        + instance.col1 * (y - 0.5);

    // Convert pixel coordinates to normalized device coordinates (NDC)
    // NDC ranges from -1 to 1, with (0,0) at the center of the viewport
    let ndc_x = pixel.x / f32(config.width) * 2.0 - 1.0;
    let ndc_y = 1.0 - pixel.y / f32(config.height) * 2.0;

    // UV in normalized texture coordinates for sampling.
    let src_pos = vec2<f32>(f32(instance.src_xy & 0xFFFFu), f32(instance.src_xy >> 16u));
    let src_size = vec2<f32>(f32(instance.src_wh & 0xFFFFu), f32(instance.src_wh >> 16u));
    let atlas_dims = vec2<f32>(textureDimensions(atlas_texture_array).xy);
    let uv = (src_pos + vec2<f32>(x, y) * src_size) / atlas_dims;

    var out: VertexOutput;
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv = uv;
    out.atlas_index = instance.atlas_index;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(atlas_texture_array, atlas_sampler, in.uv, in.atlas_index);
}
