// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Filter scaling passes.

// Downscale with a separable [1,3,3,1]/8 filter. Native bilinear sampling combines
// the horizontal and vertical passes and reduces the sample count from 16 to four.
fn downscale(
    source_texture: texture_2d<f32>,
    linear_sampler: sampler,
    position: vec4<f32>,
    source_origin: vec2<u32>,
    dest_origin: vec2<u32>,
) -> vec4<f32> {
    let frag_coord = vec2<u32>(position.xy);
    let rel = vec2<i32>(frag_coord - dest_origin);
    let source_rel = vec2<f32>(rel * 2);
    let source_texel = vec2<f32>(source_origin) + source_rel;
    let source_texture_size = vec2<f32>(textureDimensions(source_texture));

    // Shift the two sample points to produce weights [1,3,3,1].
    let lo = vec2<f32>(-0.25);
    let hi = vec2<f32>(1.25);

    let s00 = textureSampleLevel(source_texture, linear_sampler, (source_texel + vec2(lo.x, lo.y) + 0.5) / source_texture_size, 0.0);
    let s01 = textureSampleLevel(source_texture, linear_sampler, (source_texel + vec2(lo.x, hi.y) + 0.5) / source_texture_size, 0.0);
    let s10 = textureSampleLevel(source_texture, linear_sampler, (source_texel + vec2(hi.x, lo.y) + 0.5) / source_texture_size, 0.0);
    let s11 = textureSampleLevel(source_texture, linear_sampler, (source_texel + vec2(hi.x, hi.y) + 0.5) / source_texture_size, 0.0);

    return (s00 + s01 + s10 + s11) * 0.25;
}

fn upscale(
    source_texture: texture_2d<f32>,
    linear_sampler: sampler,
    position: vec4<f32>,
    source_origin: vec2<u32>,
    dest_origin: vec2<u32>,
) -> vec4<f32> {
    let frag_coord = vec2<u32>(position.xy);
    let rel = vec2<i32>(frag_coord - dest_origin);
    let source_base = vec2<f32>(rel / 2);
    let phase = vec2<f32>(rel % 2);
    let source_texture_size = vec2<f32>(textureDimensions(source_texture));

    // For even phases: 75% of current, 25% of top/left.
    // For odd phases: 75% of current, 25% of bottom/right.
    let sample_offset = select(vec2(-0.25), vec2(0.25), phase == vec2(1.0));
    let source_texel = vec2<f32>(source_origin) + source_base + sample_offset;

    return textureSampleLevel(source_texture, linear_sampler, (source_texel + 0.5) / source_texture_size, 0.0);
}
