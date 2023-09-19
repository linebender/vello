// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_encoding::{ConfigUniform, DrawMonoid, DrawTag, Monoid};

use crate::cpu_dispatch::CpuBinding;

const WG_SIZE: usize = 256;

fn draw_reduce_main(n_wg: u32, config: &ConfigUniform, scene: &[u32], reduced: &mut [DrawMonoid]) {
    let drawtag_base = config.layout.draw_tag_base;
    for i in 0..n_wg {
        let mut m = DrawMonoid::default();
        for j in 0..WG_SIZE {
            let ix = i * WG_SIZE as u32 + j as u32;
            let tag = if ix < config.layout.n_draw_objects {
                scene[(drawtag_base + ix) as usize]
            } else {
                0
            };
            m = m.combine(&DrawMonoid::new(DrawTag(tag)));
        }
        reduced[i as usize] = m;
    }
}

pub fn draw_reduce(n_wg: u32, resources: &[CpuBinding]) {
    let config = resources[0].as_typed();
    let scene = resources[1].as_slice();
    let mut reduced = resources[2].as_slice_mut();
    draw_reduce_main(n_wg, &config, &scene, &mut reduced);
}
