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

@fragment
fn fs_main(@location(0) @interpolate(flat) color: vec4<f32>) -> @location(0) vec4<f32> {
    return color;
}
