// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared quad geometry helpers.

import package::helpers::util::unpack_u16_pair;

fn rect_coverage(rect_size_packed: u32, rect_frac: u32, tex_coord: vec2<f32>) -> f32 {
    let frac = unpack4x8unorm(rect_frac);
    // Calculate how much of the pixel is actually covered by the rect.
    // We do this by simply calculating the fractions in the x and y direction, and
    // then multiplying them.
    // For (maybe?) better performance, we calculate the x and y dimension in a single
    // pass by packing everything into a vec2.
    let rect_size = vec2<f32>(unpack_u16_pair(rect_size_packed));
    let tc = tex_coord;
    // + 0.5 and -0.5 since the fragment shader positions the coordinates in the center of the pixel.
    let bottom_and_right = min(tc + 0.5, rect_size - frac.zw);
    let top_and_left = max(tc - 0.5, frac.xy);
    let a = clamp(bottom_and_right - top_and_left, vec2(0.0), vec2(1.0));
    return a.x * a.y;
}

fn quad_corner(vertex_index: u32) -> vec2<f32> {
    return vec2<f32>(
        f32(vertex_index & 1u),
        f32(vertex_index >> 1u),
    );
}

fn pixel_to_ndc(pixel: vec2<f32>, target_size: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        pixel.x * 2.0 / target_size.x - 1.0,
        1.0 - pixel.y * 2.0 / target_size.y,
    );
}
