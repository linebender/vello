// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_encoding::{Monoid, PathMonoid, RenderConfig};

const WG_SIZE: usize = 256;

pub fn pathtag_reduce(config: &RenderConfig, scene: &[u32], reduced: &mut [PathMonoid]) {
    let n = config.workgroup_counts.path_reduce.0;
    let pathtag_base = config.gpu.layout.path_tag_base;
    for i in 0..n {
        let mut m = PathMonoid::default();
        for j in 0..WG_SIZE {
            let tag = scene[(pathtag_base + i * WG_SIZE as u32) as usize + j];
            m = m.combine(&PathMonoid::new(tag));
        }
        reduced[i as usize] = m;
    }
}
