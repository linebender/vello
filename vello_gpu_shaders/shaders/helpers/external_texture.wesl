// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sampling helpers for externally bound image textures.

import package::helpers::extend::{EXTEND_PAD, extend_mode};
import package::helpers::image::{
    IMAGE_QUALITY_HIGH,
    IMAGE_QUALITY_MEDIUM,
    TINT_MODE_MULTIPLY,
    get_image_extend_modes,
    get_image_offset,
    get_image_quality,
    get_image_size,
};
@if(image_bicubic)
import package::helpers::image::cubic_weights;
import package::helpers::paint::{PAINT_TEXTURE_INDEX_MASK, load_encoded_paint_texel};

fn sample_native_image_paint(
    external_texture: texture_2d<f32>,
    external_sampler: sampler,
    opacity: u32,
    sample_xy: vec2<f32>,
) -> vec4<f32> {
    let image_opacity = f32(opacity & 0xffu) * (1.0 / 255.0);
    return image_opacity * textureSampleLevel(external_texture, external_sampler, sample_xy, 0.0);
}

fn sample_image_paint(
    encoded_paints_texture: texture_2d<u32>,
    tex_width_bits: u32,
    external_texture: texture_2d<f32>,
    paint_and_rect_flag: u32,
    sample_xy: vec2<f32>,
) -> vec4<f32> {
    let paint_tex_idx = paint_and_rect_flag & PAINT_TEXTURE_INDEX_MASK;
    let image_texel0 = load_encoded_paint_texel(
        encoded_paints_texture, tex_width_bits, paint_tex_idx, 0u,
    );
    let image_texel2 = load_encoded_paint_texel(
        encoded_paints_texture, tex_width_bits, paint_tex_idx, 2u,
    );
    let image_offset = get_image_offset(image_texel0);
    let image_size = get_image_size(image_texel0);
    let image_extend_modes = get_image_extend_modes(image_texel0);
    let image_quality = get_image_quality(image_texel0);
    let packed_tint = image_texel2.y;
    let image_tint = unpack4x8unorm(packed_tint);
    let is_multiply = image_texel2.z == TINT_MODE_MULTIPLY;
    let sample_color = sample_external_image(
        external_texture,
        image_quality,
        sample_xy,
        image_offset,
        image_size,
        image_extend_modes,
    );

    return select(
        image_tint * sample_color.a,
        sample_color * image_tint,
        is_multiply
    );
}

fn external_axis_index(coord: f32, extend_mode_id: u32, size: f32) -> i32 {
    return i32(min(extend_mode(coord, extend_mode_id, size), size - 1.0));
}

// Bilinear sampling with each tap extended within the source region.
fn external_bilinear_sample(
    external_texture: texture_2d<f32>,
    coords: vec2<f32>,
    image_offset: vec2<f32>,
    image_size: vec2<f32>,
    image_extend_modes: vec2<u32>,
) -> vec4<f32> {
    let shifted_coords = coords + vec2(0.5);
    let base_coords = floor(shifted_coords);
    let coord_frac = shifted_coords - base_coords;
    var x: vec2<i32>;
    var y: vec2<i32>;

    if all(image_extend_modes == vec2<u32>(EXTEND_PAD)) {
        x = vec2<i32>(clamp(
            base_coords.xx + vec2<f32>(-1.0, 0.0),
            vec2<f32>(0.0), vec2<f32>(image_size.x - 1.0),
        ));
        y = vec2<i32>(clamp(
            base_coords.yy + vec2<f32>(-1.0, 0.0),
            vec2<f32>(0.0), vec2<f32>(image_size.y - 1.0),
        ));
    } else {
        x = vec2<i32>(
            external_axis_index(base_coords.x - 1.0, image_extend_modes.x, image_size.x),
            external_axis_index(base_coords.x, image_extend_modes.x, image_size.x),
        );
        y = vec2<i32>(
            external_axis_index(base_coords.y - 1.0, image_extend_modes.y, image_size.y),
            external_axis_index(base_coords.y, image_extend_modes.y, image_size.y),
        );
    }
    let offset = vec2<i32>(image_offset);
    let a = textureLoad(external_texture, offset + vec2<i32>(x.x, y.x), 0);
    let b = textureLoad(external_texture, offset + vec2<i32>(x.x, y.y), 0);
    let c = textureLoad(external_texture, offset + vec2<i32>(x.y, y.x), 0);
    let d = textureLoad(external_texture, offset + vec2<i32>(x.y, y.y), 0);
    return mix(mix(a, b, coord_frac.y), mix(c, d, coord_frac.y), coord_frac.x);
}

// Bicubic sampling with each tap extended within the source region.
@if(image_bicubic)
fn external_bicubic_sample(
    external_texture: texture_2d<f32>,
    coords: vec2<f32>,
    image_offset: vec2<f32>,
    image_size: vec2<f32>,
    image_extend_modes: vec2<u32>,
) -> vec4<f32> {
    let shifted_coords = coords + vec2(0.5);
    let base_coords = floor(shifted_coords);
    let frac_coords = shifted_coords - base_coords;
    let cx = cubic_weights(frac_coords.x);
    let cy = cubic_weights(frac_coords.y);

    var x: vec4<i32>;
    var y: vec4<i32>;

    if all(image_extend_modes == vec2<u32>(EXTEND_PAD)) {
        x = vec4<i32>(clamp(
            base_coords.xxxx + vec4<f32>(-2.0, -1.0, 0.0, 1.0),
            vec4<f32>(0.0), vec4<f32>(image_size.x - 1.0),
        ));
        y = vec4<i32>(clamp(
            base_coords.yyyy + vec4<f32>(-2.0, -1.0, 0.0, 1.0),
            vec4<f32>(0.0), vec4<f32>(image_size.y - 1.0),
        ));
    } else {
        x = vec4<i32>(
            external_axis_index(base_coords.x - 2.0, image_extend_modes.x, image_size.x),
            external_axis_index(base_coords.x - 1.0, image_extend_modes.x, image_size.x),
            external_axis_index(base_coords.x, image_extend_modes.x, image_size.x),
            external_axis_index(base_coords.x + 1.0, image_extend_modes.x, image_size.x),
        );
        y = vec4<i32>(
            external_axis_index(base_coords.y - 2.0, image_extend_modes.y, image_size.y),
            external_axis_index(base_coords.y - 1.0, image_extend_modes.y, image_size.y),
            external_axis_index(base_coords.y, image_extend_modes.y, image_size.y),
            external_axis_index(base_coords.y + 1.0, image_extend_modes.y, image_size.y),
        );
    }
    let offset = vec2<i32>(image_offset);

    let s00 = textureLoad(external_texture, offset + vec2<i32>(x.x, y.x), 0);
    let s10 = textureLoad(external_texture, offset + vec2<i32>(x.y, y.x), 0);
    let s20 = textureLoad(external_texture, offset + vec2<i32>(x.z, y.x), 0);
    let s30 = textureLoad(external_texture, offset + vec2<i32>(x.w, y.x), 0);

    let s01 = textureLoad(external_texture, offset + vec2<i32>(x.x, y.y), 0);
    let s11 = textureLoad(external_texture, offset + vec2<i32>(x.y, y.y), 0);
    let s21 = textureLoad(external_texture, offset + vec2<i32>(x.z, y.y), 0);
    let s31 = textureLoad(external_texture, offset + vec2<i32>(x.w, y.y), 0);

    let s02 = textureLoad(external_texture, offset + vec2<i32>(x.x, y.z), 0);
    let s12 = textureLoad(external_texture, offset + vec2<i32>(x.y, y.z), 0);
    let s22 = textureLoad(external_texture, offset + vec2<i32>(x.z, y.z), 0);
    let s32 = textureLoad(external_texture, offset + vec2<i32>(x.w, y.z), 0);

    let s03 = textureLoad(external_texture, offset + vec2<i32>(x.x, y.w), 0);
    let s13 = textureLoad(external_texture, offset + vec2<i32>(x.y, y.w), 0);
    let s23 = textureLoad(external_texture, offset + vec2<i32>(x.z, y.w), 0);
    let s33 = textureLoad(external_texture, offset + vec2<i32>(x.w, y.w), 0);

    let row0 = cx.x * s00 + cx.y * s10 + cx.z * s20 + cx.w * s30;
    let row1 = cx.x * s01 + cx.y * s11 + cx.z * s21 + cx.w * s31;
    let row2 = cx.x * s02 + cx.y * s12 + cx.z * s22 + cx.w * s32;
    let row3 = cx.x * s03 + cx.y * s13 + cx.z * s23 + cx.w * s33;
    let result = cy.x * row0 + cy.y * row1 + cy.z * row2 + cy.w * row3;

    // Clamp alpha first, then clamp premultiplied color channels against it.
    let a = clamp(result.a, 0.0, 1.0);
    return vec4<f32>(clamp(result.rgb, vec3(0.0), vec3(a)), a);
}

fn sample_external_image(
    external_texture: texture_2d<f32>,
    quality: u32,
    coords: vec2<f32>,
    image_offset: vec2<f32>,
    image_size: vec2<f32>,
    image_extend_modes: vec2<u32>,
) -> vec4<f32> {
    @if(image_bicubic)
    if quality == IMAGE_QUALITY_HIGH {
        return external_bicubic_sample(
            external_texture,
            coords,
            image_offset,
            image_size,
            image_extend_modes,
        );
    }
    if quality == IMAGE_QUALITY_MEDIUM {
        return external_bilinear_sample(
            external_texture,
            coords,
            image_offset,
            image_size,
            image_extend_modes,
        );
    }
    return textureLoad(
        external_texture,
        vec2<i32>(image_offset) + vec2<i32>(
            external_axis_index(coords.x, image_extend_modes.x, image_size.x),
            external_axis_index(coords.y, image_extend_modes.y, image_size.y),
        ),
        0,
    );
}
