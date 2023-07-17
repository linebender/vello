// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_encoding::{ConfigUniform, Monoid, PathMonoid};

use crate::cpu_dispatch::CpuBinding;

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

pub fn pathtag_scan(n_wg: u32, resources: &[CpuBinding]) {
    let r0 = resources[0].as_buf();
    let r1 = resources[1].as_buf();
    let r2 = resources[2].as_buf();
    let mut r3 = resources[3].as_buf();
    let config = bytemuck::from_bytes(&r0);
    let scene = bytemuck::cast_slice(&r1);
    let reduced = bytemuck::cast_slice(&r2);
    let tag_monoids = bytemuck::cast_slice_mut(r3.as_mut());
    pathtag_scan_main(n_wg, config, scene, reduced, tag_monoids);
}
