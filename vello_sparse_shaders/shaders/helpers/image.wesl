// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared encoded image accessors and atlas sampling (bilinear/bicubic) helpers.

/// Image rendering quality.
const IMAGE_QUALITY_LOW: u32 = 0u;
const IMAGE_QUALITY_MEDIUM: u32 = 1u;
const IMAGE_QUALITY_HIGH: u32 = 2u;

/// Tint mode constants.
const TINT_MODE_ALPHA_MASK: u32 = 0u;
const TINT_MODE_MULTIPLY: u32 = 1u;

// Encoded image layout. Must match `GpuEncodedImage` in `vello_hybrid/src/render/common.rs`.
//
// texel0.x: image_params
//   bits 0-1: quality
//   bits 2-3: extend_x
//   bits 4-5: extend_y
// texel0.y: image_size, packed as [width:16, height:16]
// texel0.z: image_offset, packed as [x:16, y:16]
// texel0.w/texel1.x/texel1.y/texel1.z: transform matrix [a, b, c, d]
// texel1.w/texel2.x: translation [tx, ty]
// texel2.y: premultiplied tint color packed as RGBA8 unorm
// texel2.z: GPU tint mode
// texel2.w: transparent padding pixels around the image in the atlas

/// The rendering quality of the image.
fn get_image_quality(texel0: vec4<u32>) -> u32 { return texel0.x & 0x3u; }

/// The extend modes in the horizontal and vertical direction.
fn get_image_extend_modes(texel0: vec4<u32>) -> vec2<u32> {
    return vec2<u32>((texel0.x >> 2u) & 0x3u, (texel0.x >> 4u) & 0x3u);
}

/// The size of the image in pixels.
fn get_image_size(texel0: vec4<u32>) -> vec2<f32> {
    return vec2<f32>(f32(texel0.y >> 16u), f32(texel0.y & 0xFFFFu));
}

/// The offset of the image in pixels.
fn get_image_offset(texel0: vec4<u32>) -> vec2<f32> {
    return vec2<f32>(f32(texel0.z >> 16u), f32(texel0.z & 0xFFFFu));
}

/// 2x2 linear part of the affine transform (columns [a,b] and [c,d]).
fn get_image_transform(texel0: vec4<u32>, texel1: vec4<u32>) -> mat2x2<f32> {
    return mat2x2<f32>(
        vec2<f32>(bitcast<f32>(texel0.w), bitcast<f32>(texel1.x)),
        vec2<f32>(bitcast<f32>(texel1.y), bitcast<f32>(texel1.z))
    );
}

/// Translation part of the affine transform [tx, ty].
fn get_image_translate(texel1: vec4<u32>, texel2: vec4<u32>) -> vec2<f32> {
    return vec2<f32>(bitcast<f32>(texel1.w), bitcast<f32>(texel2.x));
}

// Cubic resampler logic borrowed from Skia (same as CPU cubic_resampler function)
// Mitchell-Netravali cubic filter coefficients with parameters B=1/3 and C=1/3
const MF: array<vec4<f32>, 4> = array<vec4<f32>, 4>(
    vec4<f32>(
        (1.0 / 6.0) / 3.0,
        -(3.0 / 6.0) / 3.0 - 1.0 / 3.0,
        (3.0 / 6.0) / 3.0 + 2.0 * 1.0 / 3.0,
        -(1.0 / 6.0) / 3.0 - 1.0 / 3.0
    ),
    vec4<f32>(
        1.0 - (2.0 / 6.0) / 3.0,
        0.0,
        -3.0 + (12.0 / 6.0) / 3.0 + 1.0 / 3.0,
        2.0 - (9.0 / 6.0) / 3.0 - 1.0 / 3.0
    ),
    vec4<f32>(
        (1.0 / 6.0) / 3.0,
        (3.0 / 6.0) / 3.0 + 1.0 / 3.0,
        3.0 - (15.0 / 6.0) / 3.0 - 2.0 * 1.0 / 3.0,
        -2.0 + (9.0 / 6.0) / 3.0 + 1.0 / 3.0
    ),
    vec4<f32>(
        0.0,
        0.0,
        -1.0 / 3.0,
        (1.0 / 6.0) / 3.0 + 1.0 / 3.0
    )
);

// Calculate the weights for a single fractional value (same as CPU weights function)
fn cubic_weights(fract: f32) -> vec4<f32> {
    return vec4<f32>(
        single_weight(fract, MF[0][0], MF[0][1], MF[0][2], MF[0][3]),
        single_weight(fract, MF[1][0], MF[1][1], MF[1][2], MF[1][3]),
        single_weight(fract, MF[2][0], MF[2][1], MF[2][2], MF[2][3]),
        single_weight(fract, MF[3][0], MF[3][1], MF[3][2], MF[3][3])
    );
}

// Calculate a weight based on the fractional value t and the cubic coefficients
// This matches the CPU implementation exactly
fn single_weight(t: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    return t * (t * (t * d + c) + b) + a;
}
