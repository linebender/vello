// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Copy atlas regions from scratch back into the layer atlas.

struct BlendInstance {
    @location(0) dest_origin: vec2<u32>,
    @location(1) source_origin: vec2<u32>,
    @location(2) size: vec2<u32>,
    @location(3) texture_indices: vec2<u32>,
    @location(4) blend_mode: vec2<u32>,
    @location(5) opacity: u32,
    @location(6) target_size: vec2<u32>,
    @location(7) bbox_origin: vec2<u32>,
    @location(8) source_scene_origin: vec2<u32>,
    @location(9) source_size: vec2<u32>,
}

struct VertexOutput {
    @location(0) source_xy: vec2<f32>,
    @builtin(position) position: vec4<f32>,
}

@group(1) @binding(0)
var scratch_texture: texture_2d<f32>;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    instance: BlendInstance,
) -> VertexOutput {
    let x = f32(vertex_index & 1u);
    let y = f32(vertex_index >> 1u);
    let local = vec2<f32>(x, y) * vec2<f32>(instance.size);
    let dest_xy = vec2<f32>(instance.dest_origin) + local;

    var out: VertexOutput;
    out.source_xy = vec2<f32>(instance.source_origin) + local;

    let ndc_x = dest_xy.x * 2.0 / f32(instance.target_size.x) - 1.0;
    let ndc_y = 1.0 - dest_xy.y * 2.0 / f32(instance.target_size.y);
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(
    @location(0) source_xy: vec2<f32>,
) -> @location(0) vec4<f32> {
    return textureLoad(scratch_texture, vec2<i32>(source_xy), 0);
}
