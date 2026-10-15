// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Gaussian blur filter passes.

// TODO: Add support for edge modes when blurring. This makes it harder or impossible to use
// bilinear sampling because image padding is always transparent, so only add it when needed.

const BLUR_HORIZONTAL: vec2<f32> = vec2<f32>(1.0, 0.0);
const BLUR_VERTICAL: vec2<f32> = vec2<f32>(0.0, 1.0);

// Per-axis kernels are read through plain helpers rather than a struct, see the
// `no_structs_in_fragment` lint. The x-axis kernel (also a drop shadow's only kernel)
// lives in texels 0 and 1, a blur's y-axis kernel in texels 2 and 3.

/// Number of linear taps of the x-axis kernel stored in the packed filter header.
fn get_n_linear_taps_x(texel0: vec4<u32>) -> u32 { return (texel0.x >> 11u) & 0x3u; }

/// Number of linear taps of the y-axis kernel stored in the packed filter header.
fn get_n_linear_taps_y(texel0: vec4<u32>) -> u32 { return (texel0.x >> 18u) & 0x3u; }

/// Center weight of the x-axis kernel.
fn get_center_weight_x(texel0: vec4<u32>) -> f32 { return bitcast<f32>(texel0.y); }

/// Center weight of the y-axis kernel.
fn get_center_weight_y(texel2: vec4<u32>) -> f32 { return bitcast<f32>(texel2.x); }

/// Linear sample weights of the x-axis kernel.
fn get_linear_weights_x(texel0: vec4<u32>, texel1: vec4<u32>) -> vec3<f32> {
    return vec3<f32>(
        bitcast<f32>(texel0.z),
        bitcast<f32>(texel0.w),
        bitcast<f32>(texel1.x),
    );
}

/// Linear sample weights of the y-axis kernel.
fn get_linear_weights_y(texel2: vec4<u32>) -> vec3<f32> {
    return vec3<f32>(
        bitcast<f32>(texel2.y),
        bitcast<f32>(texel2.z),
        bitcast<f32>(texel2.w),
    );
}

/// Linear sample offsets of the x-axis kernel.
fn get_linear_offsets_x(texel1: vec4<u32>) -> vec3<f32> {
    return vec3<f32>(
        bitcast<f32>(texel1.y),
        bitcast<f32>(texel1.z),
        bitcast<f32>(texel1.w),
    );
}

/// Linear sample offsets of the y-axis kernel.
fn get_linear_offsets_y(texel3: vec4<u32>) -> vec3<f32> {
    return vec3<f32>(
        bitcast<f32>(texel3.x),
        bitcast<f32>(texel3.y),
        bitcast<f32>(texel3.z),
    );
}

fn gaussian_blur(
    source_texture: texture_2d<f32>,
    linear_sampler: sampler,
    source_origin: vec2<u32>,
    source_rel: vec2<f32>,
    direction: vec2<f32>,
    center_weight: f32,
    linear_weights: vec3<f32>,
    linear_offsets: vec3<f32>,
    n_linear_taps: u32,
) -> vec4<f32> {
    let source_texel = vec2<f32>(source_origin) + source_rel;
    let source_texture_size = vec2<f32>(textureDimensions(source_texture));

    var color = textureSampleLevel(
        source_texture,
        linear_sampler,
        (source_texel + 0.5) / source_texture_size,
        0.0,
    ) * center_weight;

    if n_linear_taps > 0u {
        let distance = direction * linear_offsets.x;
        color += textureSampleLevel(source_texture, linear_sampler, (source_texel + distance + 0.5) / source_texture_size, 0.0) * linear_weights.x;
        color += textureSampleLevel(source_texture, linear_sampler, (source_texel - distance + 0.5) / source_texture_size, 0.0) * linear_weights.x;
    }
    if n_linear_taps > 1u {
        let distance = direction * linear_offsets.y;
        color += textureSampleLevel(source_texture, linear_sampler, (source_texel + distance + 0.5) / source_texture_size, 0.0) * linear_weights.y;
        color += textureSampleLevel(source_texture, linear_sampler, (source_texel - distance + 0.5) / source_texture_size, 0.0) * linear_weights.y;
    }
    if n_linear_taps > 2u {
        let distance = direction * linear_offsets.z;
        color += textureSampleLevel(source_texture, linear_sampler, (source_texel + distance + 0.5) / source_texture_size, 0.0) * linear_weights.z;
        color += textureSampleLevel(source_texture, linear_sampler, (source_texel - distance + 0.5) / source_texture_size, 0.0) * linear_weights.z;
    }

    return color;
}
