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
    get_image_size
};
@if(external_texture_ycbcr)
import package::helpers::image::{
    get_image_source_kind,
    get_image_ycbcr_matrix,
    get_image_ycbcr_range
};
@if(image_bicubic)
import package::helpers::image::cubic_weights;
import package::helpers::paint::{PAINT_TEXTURE_INDEX_MASK, load_encoded_paint_texel};
@if(external_texture_ycbcr)
import package::helpers::ycbcr::sample_external_ycbcr_nv12;

// Encoded image source kinds. Must match `vello_gpu/src/render/common.rs`.
@if(external_texture_ycbcr)
const IMAGE_SOURCE_EXTERNAL_YCBCR_NV12: u32 = 2u;

fn sample_image_paint(
    encoded_paints_texture: texture_2d<u32>,
    tex_width_bits: u32,
    external_texture: texture_2d<f32>,
    paint_and_rect_flag: u32,
    sample_xy: vec2<f32>,
) -> vec4<f32> {
    let paint_tex_idx = paint_and_rect_flag & PAINT_TEXTURE_INDEX_MASK;
    let texel0 = load_encoded_paint_texel(
            encoded_paints_texture,
            tex_width_bits,
            paint_tex_idx,
            0u,
        );
    let texel2 = load_encoded_paint_texel(
            encoded_paints_texture,
            tex_width_bits,
            paint_tex_idx,
            2u,
        );
    let color = sample_external_image(
            external_texture,
            get_image_quality(texel0),
            get_image_sample_coords(texel0, sample_xy),
            get_image_offset(texel0),
            get_image_size(texel0),
        );
    return apply_image_tint(color, texel2);
}

// Like `sample_image_paint`, but also handles NV12 sources, whose Cb/Cr plane is bound to
// `external_texture_uv`.
@if(external_texture_ycbcr)
fn sample_image_paint_with_ycbcr(
    encoded_paints_texture: texture_2d<u32>,
    tex_width_bits: u32,
    external_texture: texture_2d<f32>,
    external_texture_uv: texture_2d<f32>,
    paint_and_rect_flag: u32,
    sample_xy: vec2<f32>,
) -> vec4<f32> {
    let paint_tex_idx = paint_and_rect_flag & PAINT_TEXTURE_INDEX_MASK;
    let texel0 = load_encoded_paint_texel(
            encoded_paints_texture,
            tex_width_bits,
            paint_tex_idx,
            0u,
        );
    let texel2 = load_encoded_paint_texel(
            encoded_paints_texture,
            tex_width_bits,
            paint_tex_idx,
            2u,
        );
    let quality = get_image_quality(texel0);
    let coords = get_image_sample_coords(texel0, sample_xy);
    let offset = get_image_offset(texel0);
    let size = get_image_size(texel0);
    var color: vec4<f32>;
    if get_image_source_kind(texel0) == IMAGE_SOURCE_EXTERNAL_YCBCR_NV12 {
        color = sample_external_ycbcr_nv12(
                external_texture,
                external_texture_uv,
                quality,
                get_image_ycbcr_matrix(texel0),
                get_image_ycbcr_range(texel0),
                coords,
                offset,
                size,
            );
    } else {
        color = sample_external_image(external_texture, quality, coords, offset, size);
    }
    return apply_image_tint(color, texel2);
}

// Sample position in texture space, after applying the image's extend modes.
fn get_image_sample_coords(texel0: vec4<u32>, sample_xy: vec2<f32>) -> vec2<f32> {
    let offset = get_image_offset(texel0);
    let size = get_image_size(texel0);
    let extend_modes = get_image_extend_modes(texel0);

    // TODO: add a fast path for images where we are using bilinear sampling and want transparent pixels,
    // using GPU-native bilinear sampling

    return offset + vec2<f32>(
            extend_mode(sample_xy.x, extend_modes.x, size.x),
            extend_mode(sample_xy.y, extend_modes.y, size.y),
        );
}

fn apply_image_tint(color: vec4<f32>, texel2: vec4<u32>) -> vec4<f32> {
    let tint = unpack4x8unorm(texel2.y);
    return select(tint * color.a, color * tint, texel2.z == TINT_MODE_MULTIPLY);
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
@if(image_bicubic)
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

    let s00 = textureLoad(
            external_texture,
            vec2<i32>(clamp(coords + vec2(-1.5, -1.5), image_offset, image_max)),
            0,
        );
    let s10 = textureLoad(
            external_texture,
            vec2<i32>(clamp(coords + vec2(-0.5, -1.5), image_offset, image_max)),
            0,
        );
    let s20 = textureLoad(
            external_texture,
            vec2<i32>(clamp(coords + vec2(0.5, -1.5), image_offset, image_max)),
            0,
        );
    let s30 = textureLoad(
            external_texture,
            vec2<i32>(clamp(coords + vec2(1.5, -1.5), image_offset, image_max)),
            0,
        );

    let s01 = textureLoad(
            external_texture,
            vec2<i32>(clamp(coords + vec2(-1.5, -0.5), image_offset, image_max)),
            0,
        );
    let s11 = textureLoad(
            external_texture,
            vec2<i32>(clamp(coords + vec2(-0.5, -0.5), image_offset, image_max)),
            0,
        );
    let s21 = textureLoad(
            external_texture,
            vec2<i32>(clamp(coords + vec2(0.5, -0.5), image_offset, image_max)),
            0,
        );
    let s31 = textureLoad(
            external_texture,
            vec2<i32>(clamp(coords + vec2(1.5, -0.5), image_offset, image_max)),
            0,
        );

    let s02 = textureLoad(
            external_texture,
            vec2<i32>(clamp(coords + vec2(-1.5, 0.5), image_offset, image_max)),
            0,
        );
    let s12 = textureLoad(
            external_texture,
            vec2<i32>(clamp(coords + vec2(-0.5, 0.5), image_offset, image_max)),
            0,
        );
    let s22 = textureLoad(
            external_texture,
            vec2<i32>(clamp(coords + vec2(0.5, 0.5), image_offset, image_max)),
            0,
        );
    let s32 = textureLoad(
            external_texture,
            vec2<i32>(clamp(coords + vec2(1.5, 0.5), image_offset, image_max)),
            0,
        );

    let s03 = textureLoad(
            external_texture,
            vec2<i32>(clamp(coords + vec2(-1.5, 1.5), image_offset, image_max)),
            0,
        );
    let s13 = textureLoad(
            external_texture,
            vec2<i32>(clamp(coords + vec2(-0.5, 1.5), image_offset, image_max)),
            0,
        );
    let s23 = textureLoad(
            external_texture,
            vec2<i32>(clamp(coords + vec2(0.5, 1.5), image_offset, image_max)),
            0,
        );
    let s33 = textureLoad(
            external_texture,
            vec2<i32>(clamp(coords + vec2(1.5, 1.5), image_offset, image_max)),
            0,
        );

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
    @if(image_bicubic)
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
