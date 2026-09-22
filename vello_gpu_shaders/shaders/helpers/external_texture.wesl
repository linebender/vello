// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sampling helpers for externally bound image textures.

import package::helpers::image::{
    IMAGE_QUALITY_HIGH,
    IMAGE_QUALITY_MEDIUM,
    cubic_weights
};
import package::helpers::extend::extend_mode;

fn external_texture_load(
    external_texture: texture_2d<f32>,
    coords: vec2<f32>,
    image_offset: vec2<f32>,
    image_size: vec2<f32>,
    image_extend_modes: vec2<u32>,
) -> vec4<f32> {
    let extended_coords = min(
        vec2<f32>(
            extend_mode(coords.x, image_extend_modes.x, image_size.x),
            extend_mode(coords.y, image_extend_modes.y, image_size.y),
        ),
        image_size - vec2(1.0),
    );
    return textureLoad(
        external_texture,
        vec2<i32>(image_offset + extended_coords),
        0,
    );
}

// This is the same as `helpers::image::bilinear_sample`, but for a 2D texture
// instead of an atlas texture array.
fn external_bilinear_sample(
    external_texture: texture_2d<f32>,
    coords: vec2<f32>,
    image_offset: vec2<f32>,
    image_size: vec2<f32>,
    image_extend_modes: vec2<u32>,
) -> vec4<f32> {
    let coord_frac = fract(coords + vec2(0.5));
    let a = external_texture_load(
        external_texture,
        coords + vec2(-0.5, -0.5),
        image_offset,
        image_size,
        image_extend_modes,
    );
    let b = external_texture_load(
        external_texture,
        coords + vec2(-0.5, 0.5),
        image_offset,
        image_size,
        image_extend_modes,
    );
    let c = external_texture_load(
        external_texture,
        coords + vec2(0.5, -0.5),
        image_offset,
        image_size,
        image_extend_modes,
    );
    let d = external_texture_load(
        external_texture,
        coords + vec2(0.5, 0.5),
        image_offset,
        image_size,
        image_extend_modes,
    );
    return mix(mix(a, b, coord_frac.y), mix(c, d, coord_frac.y), coord_frac.x);
}

// This is the same as `helpers::image::bicubic_sample`, but for a 2D texture
// instead of an atlas texture array.
fn external_bicubic_sample(
    external_texture: texture_2d<f32>,
    coords: vec2<f32>,
    image_offset: vec2<f32>,
    image_size: vec2<f32>,
    image_extend_modes: vec2<u32>,
) -> vec4<f32> {
    let frac_coords = fract(coords + 0.5);
    let cx = cubic_weights(frac_coords.x);
    let cy = cubic_weights(frac_coords.y);

    let s00 = external_texture_load(external_texture, coords + vec2(-1.5, -1.5), image_offset, image_size, image_extend_modes);
    let s10 = external_texture_load(external_texture, coords + vec2(-0.5, -1.5), image_offset, image_size, image_extend_modes);
    let s20 = external_texture_load(external_texture, coords + vec2(0.5, -1.5), image_offset, image_size, image_extend_modes);
    let s30 = external_texture_load(external_texture, coords + vec2(1.5, -1.5), image_offset, image_size, image_extend_modes);

    let s01 = external_texture_load(external_texture, coords + vec2(-1.5, -0.5), image_offset, image_size, image_extend_modes);
    let s11 = external_texture_load(external_texture, coords + vec2(-0.5, -0.5), image_offset, image_size, image_extend_modes);
    let s21 = external_texture_load(external_texture, coords + vec2(0.5, -0.5), image_offset, image_size, image_extend_modes);
    let s31 = external_texture_load(external_texture, coords + vec2(1.5, -0.5), image_offset, image_size, image_extend_modes);

    let s02 = external_texture_load(external_texture, coords + vec2(-1.5, 0.5), image_offset, image_size, image_extend_modes);
    let s12 = external_texture_load(external_texture, coords + vec2(-0.5, 0.5), image_offset, image_size, image_extend_modes);
    let s22 = external_texture_load(external_texture, coords + vec2(0.5, 0.5), image_offset, image_size, image_extend_modes);
    let s32 = external_texture_load(external_texture, coords + vec2(1.5, 0.5), image_offset, image_size, image_extend_modes);

    let s03 = external_texture_load(external_texture, coords + vec2(-1.5, 1.5), image_offset, image_size, image_extend_modes);
    let s13 = external_texture_load(external_texture, coords + vec2(-0.5, 1.5), image_offset, image_size, image_extend_modes);
    let s23 = external_texture_load(external_texture, coords + vec2(0.5, 1.5), image_offset, image_size, image_extend_modes);
    let s33 = external_texture_load(external_texture, coords + vec2(1.5, 1.5), image_offset, image_size, image_extend_modes);

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
    return external_texture_load(
        external_texture,
        coords,
        image_offset,
        image_size,
        image_extend_modes,
    );
}
