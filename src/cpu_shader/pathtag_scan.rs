// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_encoding::{Monoid, PathMonoid, RenderConfig};

const WG_SIZE: usize = 256;

pub fn pathtag_scan(
    config: &RenderConfig,
    scene: &[u32],
    reduced: &[PathMonoid],
    tag_monoids: &mut [PathMonoid],
) {
    let n = config.workgroup_counts.path_scan.0;
    let pathtag_base = config.gpu.layout.path_tag_base;
    let mut prefix = PathMonoid::default();
    for i in 0..n {
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
