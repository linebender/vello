// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

use vello_encoding::{ConfigUniform, DrawMonoid, DrawTag, Monoid};

use super::{CpuBinding, util::read_draw_tag_from_scene};

const WG_SIZE: usize = 256;

fn draw_reduce_main(n_wg: u32, config: &ConfigUniform, scene: &[u32], reduced: &mut [DrawMonoid]) {
    let num_blocks_total = (config.layout.n_draw_objects as usize).div_ceil(WG_SIZE);
    let n_blocks_base = num_blocks_total / WG_SIZE;
    let remainder = num_blocks_total % WG_SIZE;
    for i in 0..n_wg as usize {
        let first_block = n_blocks_base * i + i.min(remainder);
        let n_blocks = n_blocks_base + (i < remainder) as usize;
        let mut m = DrawMonoid::default();
        for j in 0..WG_SIZE * n_blocks {
            let ix = (first_block * WG_SIZE) as u32 + j as u32;
            let tag = read_draw_tag_from_scene(config, scene, ix);
            m = m.combine(&DrawMonoid::new(DrawTag(tag)));
        }
        reduced[i] = m;
    }
}

pub fn draw_reduce(n_wg: u32, resources: &[CpuBinding<'_>]) {
    let config = resources[0].as_typed();
    let scene = resources[1].as_slice();
    let mut reduced = resources[2].as_slice_mut();
    draw_reduce_main(n_wg, &config, &scene, &mut reduced);
}
