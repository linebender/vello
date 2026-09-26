// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Anti-aliasing of rect strips.
//!
//! A rect strip reproduces the alphas the strip renderer writes for the same rectangle exactly
//! (see `vello_common::rect`), so rectangles look the same whichever way they are drawn.

// Returns the alpha (0-255) of `pixel` in a rect strip of `size` pixels.
//
// `alphas` holds the alphas of the first column, last column, first row and last row, from the
// low byte up. A corner pixel combines its column and row alphas as
// round(col_alpha * row_alpha / 255), which can be one off from the exact alpha, and adds its
// correction from `corner_corrections`: 2-bit two's complement values for the top-left,
// top-right, bottom-left and bottom-right pixel, from the low bits up.
fn rect_alpha(pixel: vec2<u32>, size: vec2<u32>, alphas: u32, corner_corrections: u32) -> u32 {
    // Interior pixels use alpha 255 and an offset of 8, which shifts the corrections out entirely;
    // only corner pixels land on one.
    var col_alpha = 255u;
    var col_offset = 8u;
    if pixel.x == 0u {
        col_alpha = alphas & 0xffu;
        col_offset = 0u;
    } else if pixel.x == size.x - 1u {
        col_alpha = (alphas >> 8u) & 0xffu;
        col_offset = 2u;
    }
    var row_alpha = 255u;
    var row_offset = 8u;
    if pixel.y == 0u {
        row_alpha = (alphas >> 16u) & 0xffu;
        row_offset = 0u;
    } else if pixel.y == size.y - 1u {
        row_alpha = alphas >> 24u;
        row_offset = 4u;
    }

    let p = col_alpha * row_alpha + 128u;
    // Maps the corrections 0b00, 0b01 and 0b11 to 1, 2 and 0.
    let biased_correction = ((corner_corrections >> (col_offset + row_offset)) + 1u) & 3u;
    return ((p + (p >> 8u)) >> 8u) + biased_correction - 1u;
}
