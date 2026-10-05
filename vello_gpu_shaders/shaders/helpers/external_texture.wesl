// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sampling helpers for externally bound image textures.

import package::helpers::extend::extend_mode;
import package::helpers::image::{
    IMAGE_QUALITY_HIGH,
    IMAGE_QUALITY_MEDIUM,
    TINT_MODE_MULTIPLY,
    get_image_extend_modes,
    get_image_offset,
    get_image_quality,
    get_image_size,
};
import package::helpers::image::cubic_weights;
import package::helpers::paint::{PAINT_TEXTURE_INDEX_MASK, load_encoded_paint_texel};

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
    let extended_xy = vec2<f32>(
        extend_mode(sample_xy.x, image_extend_modes.x, image_size.x),
        extend_mode(sample_xy.y, image_extend_modes.y, image_size.y)
    );

    // TODO: add a fast path for images where we are using bilinear sampling and want transparent pixels,
    // using GPU-native bilinear sampling

    let final_xy = image_offset + extended_xy;
    let sample_color = sample_external_image(
        external_texture,
        image_quality,
        final_xy,
        image_offset,
        image_size,
    );

    return select(
        image_tint * sample_color.a,
        sample_color * image_tint,
        is_multiply
    );
}

// This is the same as `helpers::image::bilinear_sample`, but for a 2D texture
// instead of an atlas texture array.
fn external_bilinear_sample(
    external_texture: texture_2d<f32>,
    coords: vec2<f32>,
    image_offset: vec2<f32>,
    image_size: vec2<f32>,
) -> vec4<f32> {
    let image_max = image_offset + image_size - vec2(1.0);
    let clamped_coords = clamp(coords, image_offset, image_max);
    let coord_quad = vec4(floor(clamped_coords), ceil(clamped_coords));
    let coord_frac = fract(coords);
    let a = textureLoad(external_texture, vec2<i32>(coord_quad.xy), 0);
    let b = textureLoad(external_texture, vec2<i32>(coord_quad.xw), 0);
    let c = textureLoad(external_texture, vec2<i32>(coord_quad.zy), 0);
    let d = textureLoad(external_texture, vec2<i32>(coord_quad.zw), 0);
    return mix(mix(a, b, coord_frac.y), mix(c, d, coord_frac.y), coord_frac.x);
}

// This is the same as `helpers::image::bicubic_sample`, but for a 2D texture
// instead of an atlas texture array.
fn external_bicubic_sample(
    external_texture: texture_2d<f32>,
    coords: vec2<f32>,
    image_offset: vec2<f32>,
    image_size: vec2<f32>,
) -> vec4<f32> {
    let image_max = image_offset + image_size - vec2(1.0);
    let frac_coords = fract(coords + 0.5);
    let cx = cubic_weights(frac_coords.x);
    let cy = cubic_weights(frac_coords.y);

    let s00 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(-1.5, -1.5), image_offset, image_max)), 0);
    let s10 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(-0.5, -1.5), image_offset, image_max)), 0);
    let s20 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(0.5, -1.5), image_offset, image_max)), 0);
    let s30 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(1.5, -1.5), image_offset, image_max)), 0);

    let s01 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(-1.5, -0.5), image_offset, image_max)), 0);
    let s11 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(-0.5, -0.5), image_offset, image_max)), 0);
    let s21 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(0.5, -0.5), image_offset, image_max)), 0);
    let s31 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(1.5, -0.5), image_offset, image_max)), 0);

    let s02 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(-1.5, 0.5), image_offset, image_max)), 0);
    let s12 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(-0.5, 0.5), image_offset, image_max)), 0);
    let s22 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(0.5, 0.5), image_offset, image_max)), 0);
    let s32 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(1.5, 0.5), image_offset, image_max)), 0);

    let s03 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(-1.5, 1.5), image_offset, image_max)), 0);
    let s13 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(-0.5, 1.5), image_offset, image_max)), 0);
    let s23 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(0.5, 1.5), image_offset, image_max)), 0);
    let s33 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(1.5, 1.5), image_offset, image_max)), 0);

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
) -> vec4<f32> {
    if quality == IMAGE_QUALITY_HIGH {
        return external_bicubic_sample(external_texture, coords, image_offset, image_size);
    }
    if quality == IMAGE_QUALITY_MEDIUM {
        return external_bilinear_sample(
            external_texture,
            coords - vec2(0.5),
            image_offset,
            image_size,
        );
    }
    return textureLoad(external_texture, vec2<u32>(coords), 0);
}
