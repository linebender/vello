// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

import package::helpers::geometry::{pixel_to_ndc, quad_corner};

struct ClearVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) origin: vec2<u32>,
    @location(1) size: vec2<u32>,
    @location(2) target_size: vec2<u32>,
    @location(3) color: u32,
) -> ClearVertexOutput {
    let corner = quad_corner(vertex_index);
    let pixel = vec2<f32>(origin) + corner * vec2<f32>(size);
    let ndc = pixel_to_ndc(pixel, vec2<f32>(target_size));
    return ClearVertexOutput(
        vec4<f32>(ndc, 0.0, 1.0),
        unpack4x8unorm(color),
    );
}

// This vertex shader is used for clearing atlas regions.
@vertex
fn vs_main_fullscreen(
    @builtin(vertex_index) vertex_index: u32,
) -> @builtin(position) vec4<f32> {
    // This generates a quad that covers the entire render target (hence "fullscreen"),
    // but the actual clearing region is controlled by the scissor test set on the render pass.
    // This approach is more efficient than generating region-specific geometry because:
    // 1. No vertex buffer needed - geometry generated from vertex index alone
    // 2. Same simple shader works for any region size
    // 3. Hardware scissor test efficiently clips to the desired region
    // 4. GPU rasterizer + scissor test is faster than complex vertex calculations
    //
    // Map vertex_index (0-3) to fullscreen quad corners in NDC:
    // 0 → (-1,-1), 1 → (1,-1), 2 → (-1,1), 3 → (1,1)
    let ndc = quad_corner(vertex_index) * 2.0 - vec2(1.0);
    return vec4<f32>(ndc, 0.0, 1.0);
}

@fragment
fn fs_main(@location(0) @interpolate(flat) color: vec4<f32>) -> @location(0) vec4<f32> {
    return color;
}

@fragment
fn fs_transparent(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
