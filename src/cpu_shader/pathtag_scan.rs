// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_encoding::{ConfigUniform, Monoid, PathMonoid};

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
            let tag = scene[pathtag_base as usize + ix];
            m = m.combine(&PathMonoid::new(tag));
            tag_monoids[ix] = m;
        }
        prefix = prefix.combine(&reduced[i as usize]);
    }
}
