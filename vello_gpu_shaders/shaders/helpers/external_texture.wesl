// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sampling helpers for externally bound image textures: RGBA textures, and biplanar Y'CbCr
//! texture pairs (a luma plane and a half-resolution chroma plane) converted to RGBA per sample.

import package::helpers::image::{
    IMAGE_QUALITY_HIGH,
    IMAGE_QUALITY_MEDIUM,
    cubic_weights
};

/// Y'CbCr matrices (image_params bits 8-9).
const YUV_MATRIX_BT601: u32 = 0u;
const YUV_MATRIX_BT709: u32 = 1u;
const YUV_MATRIX_BT2020: u32 = 2u;

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

/// The chroma-plane coordinate for a luma-plane coordinate of a 4:2:0 texture. Chroma texel
/// `i` is co-sited with luma sample `2i` (left siting) or centred between samples `2i` and
/// `2i + 1`; vertically it is always centred between two luma rows.
fn chroma_coords(coords: vec2<f32>, params: u32) -> vec2<f32> {
    let centered = ((params >> 12u) & 1u) == 1u;
    let x_shift = select(0.25, 0.0, centered);
    return vec2<f32>(coords.x * 0.5 + x_shift, coords.y * 0.5);
}

/// Converts texture-normalised Y' (x), Cb (y) and Cr (z) samples to opaque premultiplied RGBA
/// using the matrix, range and depth in `params` (image_params bits 8-11). Output is R'G'B' in
/// the target's encoding; BT.2020 content gets its matrix only, no transfer function.
fn yuv_to_premul_rgba(yuv: vec3<f32>, params: u32) -> vec4<f32> {
    let matrix = (params >> 8u) & 0x3u;
    let full_range = ((params >> 10u) & 1u) == 1u;
    let ten_bit = ((params >> 11u) & 1u) == 1u;

    // Code values: 8-bit samples fill the texel; 10-bit samples sit above six zero bits of a
    // 16-bit texel, so a normalised texel is `code * 64 / 65535`.
    let codes = select(yuv * 255.0, yuv * (65535.0 / 64.0), ten_bit);
    let code_max = select(255.0, 1023.0, ten_bit);
    let luma_black = select(select(16.0, 64.0, ten_bit), 0.0, full_range);
    let luma_range = select(select(219.0, 876.0, ten_bit), code_max, full_range);
    let chroma_zero = select(128.0, 512.0, ten_bit);
    let chroma_range = select(select(224.0, 896.0, ten_bit), code_max, full_range);
    let y = (codes.x - luma_black) / luma_range;
    let cb = (codes.y - chroma_zero) / chroma_range;
    let cr = (codes.z - chroma_zero) / chroma_range;

    // Luma coefficients Kr and Kb; Kg = 1 - Kr - Kb.
    let kr = select(select(0.299, 0.2126, matrix == YUV_MATRIX_BT709), 0.2627, matrix == YUV_MATRIX_BT2020);
    let kb = select(select(0.114, 0.0722, matrix == YUV_MATRIX_BT709), 0.0593, matrix == YUV_MATRIX_BT2020);
    let kg = 1.0 - kr - kb;
    let r = y + 2.0 * (1.0 - kr) * cr;
    let b = y + 2.0 * (1.0 - kb) * cb;
    let g = y - (2.0 * kb * (1.0 - kb) * cb + 2.0 * kr * (1.0 - kr) * cr) / kg;
    return vec4<f32>(clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}

/// Samples a biplanar 4:2:0 texture pair at luma-plane texel `coords` within the image at
/// `image_offset` of `image_size` luma texels: luma at the requested quality (nearest,
/// bilinear or bicubic) and chroma always bilinearly from the half-resolution plane with the
/// siting in `params`, converted to opaque premultiplied RGBA. Chroma stays bilinear at low
/// quality because a pixel-aligned image is sampled at low quality as an optimisation (nearest
/// is exact for its luma), and replicating half-resolution chroma there would show as blocks.
fn sample_external_yuv420(
    luma_texture: texture_2d<f32>,
    chroma_texture: texture_2d<f32>,
    quality: u32,
    params: u32,
    coords: vec2<f32>,
    image_offset: vec2<f32>,
    image_size: vec2<f32>,
) -> vec4<f32> {
    let luma = sample_external_image(luma_texture, quality, coords, image_offset, image_size).r;
    // The chroma plane covers the image's luma region rounded outwards to whole chroma texels.
    let chroma_offset = floor(image_offset * 0.5);
    let chroma_size = ceil((image_offset + image_size) * 0.5) - chroma_offset;
    let chroma = external_bilinear_sample(
        chroma_texture,
        chroma_coords(coords, params) - vec2(0.5),
        chroma_offset,
        chroma_size,
    ).rg;
    return yuv_to_premul_rgba(vec3<f32>(luma, chroma), params);
}
