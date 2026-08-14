// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Drop shadow filter passes.

fn sample_texture(
    texture: texture_2d<f32>,
    origin: vec2<u32>,
    rel_coord: vec2<f32>,
) -> vec4<f32> {
    let coord = vec2<u32>(vec2<i32>(origin) + vec2<i32>(rel_coord));
    return textureLoad(texture, coord, 0);
}

fn get_drop_shadow_offset(texel2: vec4<u32>) -> vec2<f32> {
    return vec2<f32>(bitcast<f32>(texel2.x), bitcast<f32>(texel2.y));
}

fn get_drop_shadow_color(texel2: vec4<u32>) -> vec4<f32> {
    return unpack4x8unorm(texel2.z);
}

fn composite_drop_shadow(
    source_texture: texture_2d<f32>,
    original_texture: texture_2d<f32>,
    source_origin: vec2<u32>,
    original_origin: vec2<u32>,
    rel_coord: vec2<f32>,
    texel2: vec4<u32>,
) -> vec4<f32> {
    let blurred = sample_texture(source_texture, source_origin, rel_coord);
    let shadow_result = get_drop_shadow_color(texel2) * blurred.a;
    let original = sample_texture(original_texture, original_origin, rel_coord);

    // Composite the original over the shadow.
    return original + shadow_result * (1.0 - original.a);
}

fn colorize_drop_shadow(
    source_texture: texture_2d<f32>,
    source_origin: vec2<u32>,
    rel_coord: vec2<f32>,
    texel2: vec4<u32>,
) -> vec4<f32> {
    let blurred = sample_texture(source_texture, source_origin, rel_coord);
    return get_drop_shadow_color(texel2) * blurred.a;
}
