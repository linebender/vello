// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Encoded paint texture access helpers.

// 24 bits for paint indices.
const PAINT_TEXTURE_INDEX_MASK: u32 = 0x00FFFFFFu;

// Convert a flat texel index to 2D texture coordinates for the encoded paints texture.
fn encoded_paint_coord(flat_idx: u32, tex_width_bits: u32) -> vec2<u32> {
    return vec2<u32>(
        flat_idx & ((1u << tex_width_bits) - 1u),
        flat_idx >> tex_width_bits
    );
}

fn load_encoded_paint_texel(
    encoded_paints_texture: texture_2d<u32>,
    tex_width_bits: u32,
    paint_tex_idx: u32,
    texel_offset: u32,
) -> vec4<u32> {
    return textureLoad(
        encoded_paints_texture,
        encoded_paint_coord(paint_tex_idx + texel_offset, tex_width_bits),
        0,
    );
}
