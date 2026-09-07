// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

use vello_encoding::{ConfigUniform, Monoid, PathMonoid};

use super::CpuBinding;

const WG_SIZE: usize = 256;

fn pathtag_scan_main(
    n_wg: u32,
    config: &ConfigUniform,
    scene: &[u32],
    reduced: &[PathMonoid],
    tag_monoids: &mut [PathMonoid],
) {
    let pathtag_base = config.layout.path_tag_base;
    let mut prefix = PathMonoid::default();
    for i in 0..n_wg {
        let mut m = prefix;
        for j in 0..WG_SIZE {
            let ix = (i * WG_SIZE as u32) as usize + j;
            tag_monoids[ix] = m;
            let tag = scene[pathtag_base as usize + ix];
            m = m.combine(&PathMonoid::new(tag));
        }
        prefix = prefix.combine(&reduced[i as usize]);
    }
}

pub fn pathtag_scan(n_wg: u32, resources: &[CpuBinding<'_>]) {
    let config = resources[0].as_typed();
    let scene = resources[1].as_slice();
    let reduced = resources[2].as_slice();
    let mut tag_monoids = resources[3].as_slice_mut();
    pathtag_scan_main(n_wg, &config, &scene, &reduced, &mut tag_monoids);
}
