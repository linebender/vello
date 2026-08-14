// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Gaussian blur filter passes.

// TODO: Add support for edge modes when blurring. This makes it harder or impossible to use
// bilinear sampling because image padding is always transparent, so only add it when needed.

const BLUR_HORIZONTAL: vec2<f32> = vec2<f32>(1.0, 0.0);
const BLUR_VERTICAL: vec2<f32> = vec2<f32>(0.0, 1.0);

/// Number of linear taps stored in the packed filter header.
fn get_n_linear_taps(texel0: vec4<u32>) -> u32 { return (texel0.x >> 11u) & 0x3u; }

/// Center weight for gaussian blur convolution.
fn get_center_weight(texel0: vec4<u32>) -> f32 { return bitcast<f32>(texel0.y); }

/// Linear sample weights for gaussian blur convolution.
fn get_linear_weights(texel0: vec4<u32>, texel1: vec4<u32>) -> vec3<f32> {
    return vec3<f32>(
        bitcast<f32>(texel0.z),
        bitcast<f32>(texel0.w),
        bitcast<f32>(texel1.x),
    );
}

/// Linear sample offsets for gaussian blur convolution.
fn get_linear_offsets(texel1: vec4<u32>) -> vec3<f32> {
    return vec3<f32>(
        bitcast<f32>(texel1.y),
        bitcast<f32>(texel1.z),
        bitcast<f32>(texel1.w),
    );
}

fn gaussian_blur(
    source_texture: texture_2d<f32>,
    linear_sampler: sampler,
    source_origin: vec2<u32>,
    source_rel: vec2<f32>,
    direction: vec2<f32>,
    texel0: vec4<u32>,
    texel1: vec4<u32>,
) -> vec4<f32> {
    let source_texel = vec2<f32>(source_origin) + source_rel;
    let source_texture_size = vec2<f32>(textureDimensions(source_texture));
    let n_linear_taps = get_n_linear_taps(texel0);
    let center_weight = get_center_weight(texel0);

    // Use `textureSampleLevel` instead of `textureSample` because the loop below has a dynamic
    // iteration count and must work properly in the Direct3D backend.
    var color = textureSampleLevel(
        source_texture,
        linear_sampler,
        (source_texel + 0.5) / source_texture_size,
        0.0,
    ) * center_weight;

    // See https://github.com/linebender/vello/pull/1601#issuecomment-4323170395 for why we convert
    // into arrays first. See https://github.com/linebender/vello/pull/1605 for why we assign each
    // field separately.
    var weights: array<f32, 3>;
    let linear_weights = get_linear_weights(texel0, texel1);
    weights[0] = linear_weights.x;
    weights[1] = linear_weights.y;
    weights[2] = linear_weights.z;

    var offsets: array<f32, 3>;
    let linear_offsets = get_linear_offsets(texel1);
    offsets[0] = linear_offsets.x;
    offsets[1] = linear_offsets.y;
    offsets[2] = linear_offsets.z;

    for (var i = 0u; i < n_linear_taps; i++) {
        let weight = weights[i];
        let distance = direction * offsets[i];
        color += textureSampleLevel(source_texture, linear_sampler, (source_texel + distance + 0.5) / source_texture_size, 0.0) * weight;
        color += textureSampleLevel(source_texture, linear_sampler, (source_texel - distance + 0.5) / source_texture_size, 0.0) * weight;
    }

    return color;
}
