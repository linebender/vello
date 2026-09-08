// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared quad geometry helpers.

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
