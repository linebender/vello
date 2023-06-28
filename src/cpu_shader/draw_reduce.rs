// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_encoding::{DrawMonoid, DrawTag, Monoid, RenderConfig};

const WG_SIZE: usize = 256;

pub fn draw_reduce(config: &RenderConfig, scene: &[u32], reduced: &mut [DrawMonoid]) {
    let n = config.workgroup_counts.path_reduce.0;
    let drawtag_base = config.gpu.layout.draw_tag_base;
    for i in 0..n {
        let mut m = DrawMonoid::default();
        for j in 0..WG_SIZE {
            let tag = scene[(drawtag_base + i * WG_SIZE as u32) as usize + j];
            m = m.combine(&DrawMonoid::new(DrawTag(tag)));
        }
        reduced[i as usize] = m;
    }
}
