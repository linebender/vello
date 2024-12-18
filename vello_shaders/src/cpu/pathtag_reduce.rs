// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

use vello_encoding::{ConfigUniform, Monoid, PathMonoid};

use super::CpuBinding;

const WG_SIZE: usize = 256;

fn pathtag_reduce_main(
    n_wg: u32,
    config: &ConfigUniform,
    scene: &[u32],
    reduced: &mut [PathMonoid],
) {
    let pathtag_base = config.layout.path_tag_base;
    for i in 0..n_wg {
        let mut m = PathMonoid::default();
        for j in 0..WG_SIZE {
            let tag = scene[(pathtag_base + i * WG_SIZE as u32) as usize + j];
            m = m.combine(&PathMonoid::new(tag));
        }
        reduced[i as usize] = m;
    }
}

pub fn pathtag_reduce(n_wg: u32, resources: &[CpuBinding<'_>]) {
    let config = resources[0].as_typed();
    let scene = resources[1].as_slice();
    let mut reduced = resources[2].as_slice_mut();
    pathtag_reduce_main(n_wg, &config, &scene, &mut reduced);
}
