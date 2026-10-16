// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sampling helpers for externally bound image textures.

@if(full_image_support || texture_external)
import package::helpers::extend::{EXTEND_PAD, extend_mode};
@if(full_image_support)
import package::helpers::image::{
    IMAGE_QUALITY_HIGH,
    TINT_MODE_MULTIPLY,
    get_image_extend_modes,
    get_image_offset,
    get_image_quality,
    get_image_size
};
@if(full_image_support || texture_external)
import package::helpers::image::IMAGE_QUALITY_MEDIUM;
@if(full_image_support)
import package::helpers::image::cubic_weights;
@if(full_image_support)
import package::helpers::paint::{PAINT_TEXTURE_INDEX_MASK, load_encoded_paint_texel};

// With `texture_external`, the slot is a `wgpu::ExternalTexture` (or a plain RGBA view bound in
// its place) and wgpu performs any YCbCr → RGB conversion, cropping and color-space transform
// at load time.
@if(texture_external)
alias ExternalTexture = texture_external;
@if(!texture_external)
alias ExternalTexture = texture_2d<f32>;

fn load_external_texel(external_texture: ExternalTexture, coords: vec2<i32>) -> vec4<f32> {
    @if(texture_external) {
        return textureLoad(external_texture, coords);
    } @else {
        return textureLoad(external_texture, coords, 0);
    }
}

// wgpu does not clamp the YCbCr → RGB conversion, so out-of-range source samples (e.g.
// limited-range super-whites) would otherwise leave the premultiplied range.
fn clamp_external_texel(color: vec4<f32>) -> vec4<f32> {
    @if(texture_external) {
        return vec4<f32>(clamp(color.rgb, vec3(0.0), vec3(color.a)), color.a);
    } @else {
        return color;
    }
}

// `payload` holds the opacity in bits 0-7 and the native sampler params in bits 8-13.
fn sample_native_image_paint(
    external_texture: ExternalTexture,
    external_sampler: sampler,
    payload: u32,
    sample_xy: vec2<f32>,
) -> vec4<f32> {
    let image_opacity = f32(payload & 0xffu) * (1.0 / 255.0);
    @if(texture_external) {
        let color = sample_native_texture_external(
                external_texture,
                external_sampler,
                payload >> 8u,
                sample_xy,
            );
        return image_opacity * clamp_external_texel(color);
    } @else {
        return image_opacity
            * textureSampleLevel(external_texture, external_sampler, sample_xy, 0.0);
    }
}

// `texture_external` can only be sampled with clamp-to-edge addressing, so non-pad extend modes
// are emulated by filtering texel loads.
@if(texture_external)
fn sample_native_texture_external(
    external_texture: texture_external,
    external_sampler: sampler,
    sampler_params: u32,
    uv: vec2<f32>,
) -> vec4<f32> {
    let extend_modes = vec2<u32>((sampler_params >> 2u) & 0x3u, (sampler_params >> 4u) & 0x3u);
    if all(extend_modes == vec2<u32>(EXTEND_PAD)) {
        return textureSampleBaseClampToEdge(external_texture, external_sampler, uv);
    }

    let size = vec2<f32>(textureDimensions(external_texture));
    let coords = uv * size;
    if (sampler_params & 0x3u) != IMAGE_QUALITY_MEDIUM {
        return textureLoad(
                external_texture,
                vec2<i32>(
                    external_axis_index(coords.x, extend_modes.x, size.x),
                    external_axis_index(coords.y, extend_modes.y, size.y),
                ),
            );
    }

    let shifted_coords = coords - vec2(0.5);
    let base_coords = floor(shifted_coords);
    let coord_frac = shifted_coords - base_coords;
    let x = vec2<i32>(
            external_axis_index(base_coords.x, extend_modes.x, size.x),
            external_axis_index(base_coords.x + 1.0, extend_modes.x, size.x),
        );
    let y = vec2<i32>(
            external_axis_index(base_coords.y, extend_modes.y, size.y),
            external_axis_index(base_coords.y + 1.0, extend_modes.y, size.y),
        );
    let a = textureLoad(external_texture, vec2<i32>(x.x, y.x));
    let b = textureLoad(external_texture, vec2<i32>(x.x, y.y));
    let c = textureLoad(external_texture, vec2<i32>(x.y, y.x));
    let d = textureLoad(external_texture, vec2<i32>(x.y, y.y));
    return mix(mix(a, b, coord_frac.y), mix(c, d, coord_frac.y), coord_frac.x);
}

@if(full_image_support)
fn sample_image_paint(
    encoded_paints_texture: texture_2d<u32>,
    tex_width_bits: u32,
    external_texture: ExternalTexture,
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
            sample_xy,
            get_image_offset(texel0),
            get_image_size(texel0),
            get_image_extend_modes(texel0),
        );
    return apply_image_tint(clamp_external_texel(color), texel2);
}

@if(full_image_support)
fn apply_image_tint(color: vec4<f32>, texel2: vec4<u32>) -> vec4<f32> {
    let tint = unpack4x8unorm(texel2.y);
    return select(tint * color.a, color * tint, texel2.z == TINT_MODE_MULTIPLY);
}

@if(full_image_support || texture_external)
fn external_axis_index(coord: f32, extend_mode_id: u32, size: f32) -> i32 {
    return i32(min(extend_mode(coord, extend_mode_id, size), size - 1.0));
}

// Bilinear sampling with each tap extended within the source region.
@if(full_image_support)
fn external_bilinear_sample(
    external_texture: ExternalTexture,
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
        x = vec2<i32>(
                clamp(
                    base_coords.xx + vec2<f32>(-1.0, 0.0),
                    vec2<f32>(0.0),
                    vec2<f32>(image_size.x - 1.0),
                ),
            );
        y = vec2<i32>(
                clamp(
                    base_coords.yy + vec2<f32>(-1.0, 0.0),
                    vec2<f32>(0.0),
                    vec2<f32>(image_size.y - 1.0),
                ),
            );
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
    let a = load_external_texel(external_texture, offset + vec2<i32>(x.x, y.x));
    let b = load_external_texel(external_texture, offset + vec2<i32>(x.x, y.y));
    let c = load_external_texel(external_texture, offset + vec2<i32>(x.y, y.x));
    let d = load_external_texel(external_texture, offset + vec2<i32>(x.y, y.y));
    return mix(mix(a, b, coord_frac.y), mix(c, d, coord_frac.y), coord_frac.x);
}

// Bicubic sampling with each tap extended within the source region.
@if(full_image_support)
fn external_bicubic_sample(
    external_texture: ExternalTexture,
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
        x = vec4<i32>(
                clamp(
                    base_coords.xxxx + vec4<f32>(-2.0, -1.0, 0.0, 1.0),
                    vec4<f32>(0.0),
                    vec4<f32>(image_size.x - 1.0),
                ),
            );
        y = vec4<i32>(
                clamp(
                    base_coords.yyyy + vec4<f32>(-2.0, -1.0, 0.0, 1.0),
                    vec4<f32>(0.0),
                    vec4<f32>(image_size.y - 1.0),
                ),
            );
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

    let s00 = load_external_texel(external_texture, offset + vec2<i32>(x.x, y.x));
    let s10 = load_external_texel(external_texture, offset + vec2<i32>(x.y, y.x));
    let s20 = load_external_texel(external_texture, offset + vec2<i32>(x.z, y.x));
    let s30 = load_external_texel(external_texture, offset + vec2<i32>(x.w, y.x));

    let s01 = load_external_texel(external_texture, offset + vec2<i32>(x.x, y.y));
    let s11 = load_external_texel(external_texture, offset + vec2<i32>(x.y, y.y));
    let s21 = load_external_texel(external_texture, offset + vec2<i32>(x.z, y.y));
    let s31 = load_external_texel(external_texture, offset + vec2<i32>(x.w, y.y));

    let s02 = load_external_texel(external_texture, offset + vec2<i32>(x.x, y.z));
    let s12 = load_external_texel(external_texture, offset + vec2<i32>(x.y, y.z));
    let s22 = load_external_texel(external_texture, offset + vec2<i32>(x.z, y.z));
    let s32 = load_external_texel(external_texture, offset + vec2<i32>(x.w, y.z));

    let s03 = load_external_texel(external_texture, offset + vec2<i32>(x.x, y.w));
    let s13 = load_external_texel(external_texture, offset + vec2<i32>(x.y, y.w));
    let s23 = load_external_texel(external_texture, offset + vec2<i32>(x.z, y.w));
    let s33 = load_external_texel(external_texture, offset + vec2<i32>(x.w, y.w));

    let row0 = cx.x * s00 + cx.y * s10 + cx.z * s20 + cx.w * s30;
    let row1 = cx.x * s01 + cx.y * s11 + cx.z * s21 + cx.w * s31;
    let row2 = cx.x * s02 + cx.y * s12 + cx.z * s22 + cx.w * s32;
    let row3 = cx.x * s03 + cx.y * s13 + cx.z * s23 + cx.w * s33;
    let result = cy.x * row0 + cy.y * row1 + cy.z * row2 + cy.w * row3;

    // Clamp alpha first, then clamp premultiplied color channels against it.
    let a = clamp(result.a, 0.0, 1.0);
    return vec4<f32>(clamp(result.rgb, vec3(0.0), vec3(a)), a);
}

@if(full_image_support)
fn sample_external_image(
    external_texture: ExternalTexture,
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
    return load_external_texel(
            external_texture,
            vec2<i32>(image_offset) + vec2<i32>(
                external_axis_index(coords.x, image_extend_modes.x, image_size.x),
                external_axis_index(coords.y, image_extend_modes.y, image_size.y),
            ),
        );
}
