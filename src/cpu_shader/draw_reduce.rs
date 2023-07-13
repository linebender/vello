// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_encoding::{ConfigUniform, DrawMonoid, DrawTag, Monoid};

const WG_SIZE: usize = 256;

pub fn draw_reduce_main(
    n_wg: u32,
    config: &ConfigUniform,
    scene: &[u32],
    reduced: &mut [DrawMonoid],
) {
    let drawtag_base = config.layout.draw_tag_base;
    for i in 0..n_wg {
        let mut m = DrawMonoid::default();
        for j in 0..WG_SIZE {
            let tag = scene[(drawtag_base + i * WG_SIZE as u32) as usize + j];
            m = m.combine(&DrawMonoid::new(DrawTag(tag)));
        }
        reduced[i as usize] = m;
    }
}
