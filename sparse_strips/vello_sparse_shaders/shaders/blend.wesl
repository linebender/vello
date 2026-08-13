// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Blend two rendered layer atlas regions.

import package::helpers::alpha::unpack_alphas_from_channel;
import package::helpers::blend::blend_mix_compose;
import package::helpers::util::unpack_u16_pair;

const BLEND_HAS_ALPHA = 1u << 26u;

struct BlendInstance {
    @location(0) geometry_origin: u32,
    @location(1) geometry_size: u32,
    @location(2) geometry_alpha_col_idx: u32,
    @location(3) parent_texture_size: u32,
    @location(4) child_texture_origin: u32,
    @location(5) child_parent_origin: u32,
    @location(6) child_rect_size: u32,
    @location(7) blend_config: u32,
}

struct VertexOutput {
    @location(0) parent_texture_xy: vec2<f32>,
    @location(1) child_texture_xy: vec2<f32>,
    @location(2) child_local: vec2<f32>,
    @location(3) @interpolate(flat) child_rect_size: u32,
    @location(4) @interpolate(flat) blend_config: u32,
    @location(5) tex_coord: vec2<f32>,
    @builtin(position) position: vec4<f32>,
}

@group(0) @binding(0)
var layer_texture_0: texture_2d<f32>;
@group(0) @binding(1)
var layer_texture_1: texture_2d<f32>;
@group(0) @binding(2)
var alphas_texture: texture_2d<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    instance: BlendInstance,
) -> VertexOutput {
    let geometry_origin = unpack_u16_pair(instance.geometry_origin);
    let geometry_size = vec2<f32>(unpack_u16_pair(instance.geometry_size));
    let parent_texture_size = unpack_u16_pair(instance.parent_texture_size);
    let child_parent_origin = unpack_u16_pair(instance.child_parent_origin);
    let child_texture_origin = unpack_u16_pair(instance.child_texture_origin);

    let x = f32(vertex_index & 1u);
    let y = f32(vertex_index >> 1u);
    let parent_texture_xy = vec2<f32>(geometry_origin) + vec2<f32>(x, y) * geometry_size;
    let child_local = parent_texture_xy - vec2<f32>(child_parent_origin);

    var out: VertexOutput;
    out.parent_texture_xy = parent_texture_xy;
    out.child_texture_xy = vec2<f32>(child_texture_origin) + child_local;
    out.child_local = child_local;
    out.child_rect_size = instance.child_rect_size;
    out.blend_config = instance.blend_config;
    out.tex_coord = vec2<f32>(
        f32(instance.geometry_alpha_col_idx) + x * geometry_size.x,
        y * geometry_size.y,
    );

    let ndc_x = parent_texture_xy.x * 2.0 / f32(parent_texture_size.x) - 1.0;
    let ndc_y = 1.0 - parent_texture_xy.y * 2.0 / f32(parent_texture_size.y);
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    return out;
}

fn load_layer(texture_index: u32, xy: vec2<i32>) -> vec4<f32> {
    if texture_index == 0u {
        return textureLoad(layer_texture_0, xy, 0);
    } else {
        return textureLoad(layer_texture_1, xy, 0);
    }
}

@fragment
fn fs_main(
    @location(0) parent_texture_xy: vec2<f32>,
    @location(1) child_texture_xy: vec2<f32>,
    @location(2) child_local: vec2<f32>,
    @location(3) @interpolate(flat) packed_child_rect_size: u32,
    @location(4) @interpolate(flat) blend_config: u32,
    @location(5) tex_coord: vec2<f32>,
) -> @location(0) vec4<f32> {
    let child_rect_size = unpack_u16_pair(packed_child_rect_size);
    let compose = blend_config & 0xffu;
    let mix_mode = (blend_config >> 8u) & 0xffu;
    let opacity = (blend_config >> 16u) & 0xffu;
    let parent_texture_index = (blend_config >> 24u) & 1u;
    let child_texture_index = (blend_config >> 25u) & 1u;
    let has_alpha = (blend_config >> 26u) & 1u;

    var clip_coverage = 1.0;

    // We are blending with a clip path, sample the alpha value.
    if (has_alpha) != 0u {
        let alpha_index = u32(floor(tex_coord.x));
        let y = u32(floor(tex_coord.y));
        let texel_index = alpha_index / 4u;
        let channel_index = alpha_index % 4u;
        let texture_width = textureDimensions(alphas_texture).x;
        let packed = textureLoad(
            alphas_texture,
            vec2<u32>(texel_index % texture_width, texel_index / texture_width),
            0,
        );
        let column = unpack_alphas_from_channel(packed, channel_index);
        clip_coverage = f32((column >> (y * 8u)) & 0xffu) * (1.0 / 255.0);
    }

    let backdrop = load_layer(parent_texture_index, vec2<i32>(parent_texture_xy));
    let child_opacity = f32(opacity) * (1.0 / 255.0);

    var child = vec4<f32>(0.0);
    // Since we are on an atlas, make sure to explicitly set to 0 if we are sampling outside the
    // bounds of the child!
    if child_local.x >= 0.0
        && child_local.y >= 0.0
        && child_local.x < f32(child_rect_size.x)
        && child_local.y < f32(child_rect_size.y)
    {
        child = child_opacity * load_layer(child_texture_index, vec2<i32>(child_texture_xy));
    }

    let composed = blend_mix_compose(backdrop, child, compose, mix_mode);

    return mix(backdrop, composed, clip_coverage);
}
