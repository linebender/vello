// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// A minimal WGSL shader for instanced texture blitting.
//
// Each instance represents an axis-aligned rectangle that copies a region from
// the image atlas directly to the screen. This bypasses the full sparse strip
// pipeline for simple image rectangle fills.
//
// Instance data (`BlitInstance`) uses packed u16 pairs in u32 fields for minimal
// bandwidth. The vertex shader unpacks and converts to f32/NDC.

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
    /// Packed screen position: x | (y << 16)
    @location(0) dst_xy: u32,
    /// Packed screen size: w | (h << 16)
    @location(1) dst_wh: u32,
    /// Packed atlas source offset: u | (v << 16)
    @location(2) src_xy: u32,
    /// Packed atlas source size: w | (h << 16)
    @location(3) src_wh: u32,
    /// Atlas layer index
    @location(4) atlas_index: u32,
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

/// Unpack a u32 containing two packed u16 values into a vec2<f32> (unsigned).
fn unpack_u16_pair(packed: u32) -> vec2<f32> {
    return vec2<f32>(
        f32(packed & 0xFFFFu),
        f32(packed >> 16u),
    );
}

/// Unpack a u32 containing two packed i16 values (as u16 bit patterns) into a vec2<f32> (signed).
fn unpack_i16_pair(packed: u32) -> vec2<f32> {
    let lo = packed & 0xFFFFu;
    let hi = packed >> 16u;
    // Reinterpret u16 as i16 via two's complement: if bit 15 is set, subtract 65536.
    let sx = f32(lo) - select(0.0, 65536.0, lo >= 0x8000u);
    let sy = f32(hi) - select(0.0, 65536.0, hi >= 0x8000u);
    return vec2<f32>(sx, sy);
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    instance: BlitInstance,
) -> VertexOutput {
    // Triangle strip: 0=(0,0), 1=(1,0), 2=(0,1), 3=(1,1)
    let x = f32(vertex_index & 1u);
    let y = f32((vertex_index >> 1u) & 1u);

    let dst_pos = unpack_i16_pair(instance.dst_xy);
    let dst_size = unpack_u16_pair(instance.dst_wh);
    let src_pos = unpack_u16_pair(instance.src_xy);
    let src_size = unpack_u16_pair(instance.src_wh);

    // Screen pixel position of this vertex.
    let pixel_x = dst_pos.x + x * dst_size.x;
    let pixel_y = dst_pos.y + y * dst_size.y;

    // Convert to NDC: [-1, 1] range.
    let ndc_x = pixel_x / f32(config.width) * 2.0 - 1.0;
    let ndc_y = 1.0 - pixel_y / f32(config.height) * 2.0;

    // UV in normalized texture coordinates for sampling.
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
