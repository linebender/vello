// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Copy rectangular regions between intermediate textures.

struct CopyInstance {
    @location(0) target_texture_origin: u32,
    @location(1) source_texture_origin: u32,
    @location(2) copy_rect_size: u32,
    @location(3) target_texture_size: u32,
}

struct VertexOutput {
    @location(0) source_texture_xy: vec2<f32>,
    @builtin(position) position: vec4<f32>,
}

@group(1) @binding(0)
var source_texture: texture_2d<f32>;

fn unpack_u16_pair(value: u32) -> vec2<u32> {
    return vec2<u32>(value & 0xffffu, value >> 16u);
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    instance: CopyInstance,
) -> VertexOutput {
    let target_texture_origin = unpack_u16_pair(instance.target_texture_origin);
    let source_texture_origin = unpack_u16_pair(instance.source_texture_origin);
    let copy_rect_size = unpack_u16_pair(instance.copy_rect_size);
    let target_texture_size = unpack_u16_pair(instance.target_texture_size);

    let x = f32(vertex_index & 1u);
    let y = f32(vertex_index >> 1u);
    let local = vec2<f32>(x, y) * vec2<f32>(copy_rect_size);
    let target_texture_xy = vec2<f32>(target_texture_origin) + local;

    var out: VertexOutput;
    out.source_texture_xy = vec2<f32>(source_texture_origin) + local;

    let ndc_x = target_texture_xy.x * 2.0 / f32(target_texture_size.x) - 1.0;
    let ndc_y = 1.0 - target_texture_xy.y * 2.0 / f32(target_texture_size.y);
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(
    @location(0) source_texture_xy: vec2<f32>,
) -> @location(0) vec4<f32> {
    return textureLoad(source_texture, vec2<i32>(source_texture_xy), 0);
}
