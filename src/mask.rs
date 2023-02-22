// Copyright 2022 The Vello authors
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
    for i in 0..8 {
        let mut y = (i as f64 + 0.5) * 0.125;
        let x = (PATTERN[i] as f64 + 0.5) * 0.125;
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
/// The table is organized into two blocks each with MASK_HEIGHT/2 slopes.
/// The first block is negative slopes (x decreases as y increates),
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
