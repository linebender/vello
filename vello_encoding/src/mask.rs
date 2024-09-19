// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Create a lookup table of half-plane sample masks.

// Width is number of discrete translations
const MASK_WIDTH: usize = 32;
// Height is the number of discrete slopes
const MASK_HEIGHT: usize = 32;

const PATTERN: [u8; 8] = [0, 5, 3, 7, 1, 4, 6, 2];

fn one_mask(slope: f64, mut translation: f64, is_pos: bool) -> u8 {
    if is_pos {
        translation = 1. - translation;
    }
    let mut result = 0;
    for (i, item) in PATTERN.iter().enumerate() {
        let mut y = (i as f64 + 0.5) * 0.125;
        let x = (*item as f64 + 0.5) * 0.125;
        if !is_pos {
            y = 1. - y;
        }
        if (x - (1.0 - translation)) * (1. - slope) - (y - translation) * slope >= 0. {
            result |= 1 << i;
        }
    }
    result
}

/// Make a lookup table of half-plane masks.
///
/// The table is organized into two blocks each with `MASK_HEIGHT/2` slopes.
/// The first block is negative slopes (x decreases as y increases),
/// the second as positive.
pub fn make_mask_lut() -> Vec<u8> {
    (0..MASK_WIDTH * MASK_HEIGHT)
        .map(|i| {
            const HALF_HEIGHT: usize = MASK_HEIGHT / 2;
            let u = i % MASK_WIDTH;
            let v = i / MASK_WIDTH;
            let is_pos = v >= HALF_HEIGHT;
            let y = ((v % HALF_HEIGHT) as f64 + 0.5) * (1.0 / HALF_HEIGHT as f64);
            let x = (u as f64 + 0.5) * (1.0 / MASK_WIDTH as f64);
            one_mask(y, x, is_pos)
        })
        .collect()
}

// Width is number of discrete translations
const MASK16_WIDTH: usize = 64;
// Height is the number of discrete slopes
const MASK16_HEIGHT: usize = 64;

// This is based on the [D3D11 standard sample pattern].
//
// [D3D11 standard sample pattern]: https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ne-d3d11-d3d11_standard_multisample_quality_levels
const PATTERN_16: [u8; 16] = [1, 8, 4, 11, 15, 7, 3, 12, 0, 9, 5, 13, 2, 10, 6, 14];

fn one_mask_16(slope: f64, mut translation: f64, is_pos: bool) -> u16 {
    if is_pos {
        translation = 1. - translation;
    }
    let mut result = 0;
    for (i, item) in PATTERN_16.iter().enumerate() {
        let mut y = (i as f64 + 0.5) * 0.0625;
        let x = (*item as f64 + 0.5) * 0.0625;
        if !is_pos {
            y = 1. - y;
        }
        if (x - (1.0 - translation)) * (1. - slope) - (y - translation) * slope >= 0. {
            result |= 1 << i;
        }
    }
    result
}

/// Make a lookup table of half-plane masks.
///
/// The table is organized into two blocks each with `MASK16_HEIGHT/2` slopes.
/// The first block is negative slopes (x decreases as y increases),
/// the second as positive.
pub fn make_mask_lut_16() -> Vec<u8> {
    let v16 = (0..MASK16_WIDTH * MASK16_HEIGHT)
        .map(|i| {
            const HALF_HEIGHT: usize = MASK16_HEIGHT / 2;
            let u = i % MASK16_WIDTH;
            let v = i / MASK16_WIDTH;
            let is_pos = v >= HALF_HEIGHT;
            let y = ((v % HALF_HEIGHT) as f64 + 0.5) * (1.0 / HALF_HEIGHT as f64);
            let x = (u as f64 + 0.5) * (1.0 / MASK16_WIDTH as f64);
            one_mask_16(y, x, is_pos)
        })
        .collect::<Vec<_>>();
    // This annoyingly makes another copy. We can avoid that by pushing two
    // bytes per iteration of the above loop.
    bytemuck::cast_slice(&v16).into()
}
