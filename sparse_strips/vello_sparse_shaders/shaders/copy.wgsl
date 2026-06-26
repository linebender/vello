// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Copy atlas regions from scratch back into the layer atlas.

struct CopyInstance {
    @location(0) dest_origin: u32,
    @location(1) source_origin: u32,
    @location(2) size: u32,
    @location(3) target_size: u32,
}

struct VertexOutput {
    @location(0) source_xy: vec2<f32>,
    @builtin(position) position: vec4<f32>,
}

@group(1) @binding(0)
var scratch_texture: texture_2d<f32>;

fn unpack_u16_pair(value: u32) -> vec2<u32> {
    return vec2<u32>(value & 0xffffu, value >> 16u);
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    instance: CopyInstance,
) -> VertexOutput {
    let dest_origin = unpack_u16_pair(instance.dest_origin);
    let source_origin = unpack_u16_pair(instance.source_origin);
    let size = unpack_u16_pair(instance.size);
    let target_size = unpack_u16_pair(instance.target_size);

    let x = f32(vertex_index & 1u);
    let y = f32(vertex_index >> 1u);
    let local = vec2<f32>(x, y) * vec2<f32>(size);
    let dest_xy = vec2<f32>(dest_origin) + local;

    var out: VertexOutput;
    out.source_xy = vec2<f32>(source_origin) + local;

    let ndc_x = dest_xy.x * 2.0 / f32(target_size.x) - 1.0;
    let ndc_y = 1.0 - dest_xy.y * 2.0 / f32(target_size.y);
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(
    @location(0) source_xy: vec2<f32>,
) -> @location(0) vec4<f32> {
    return textureLoad(scratch_texture, vec2<i32>(source_xy), 0);
}
