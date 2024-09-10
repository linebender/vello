// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

use vello_encoding::{ConfigUniform, Monoid, PathMonoid};

use super::CpuBinding;

const WG_SIZE: usize = 256;

fn pathtag_scan_single_main(
    n_wg: u32,
    config: &ConfigUniform,
    scene: &[u32],
    tag_monoids: &mut [PathMonoid],
) {
    let size = n_wg * (WG_SIZE as u32);
    let pathtag_base = config.layout.path_tag_base;
    let mut prefix = PathMonoid::default();
    for i in 0..size {
        tag_monoids[i as usize] = prefix;
        prefix = prefix.combine(&PathMonoid::new(scene[(pathtag_base + i) as usize]));
    }
}

pub fn pathtag_scan_single(n_wg: u32, resources: &[CpuBinding]) {
    let config = resources[0].as_typed();
    let scene = resources[1].as_slice();
    let mut tag_monoids = resources[3].as_slice_mut();
    pathtag_scan_single_main(n_wg, &config, &scene, &mut tag_monoids);
}