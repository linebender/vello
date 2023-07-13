// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_encoding::{ConfigUniform, Monoid, PathMonoid};

use crate::cpu_dispatch::CpuResourceRef;

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

pub fn pathtag_reduce(n_wg: u32, resources: Vec<CpuResourceRef>) {
    let mut r_iter = resources.into_iter();
    let mut r0 = r_iter.next().unwrap();
    let config = bytemuck::from_bytes(r0.as_buf());
    let mut r1 = r_iter.next().unwrap();
    let scene = bytemuck::cast_slice(r1.as_buf());
    let mut r2 = r_iter.next().unwrap();
    let reduced = bytemuck::cast_slice_mut(r2.as_buf());
    pathtag_reduce_main(n_wg, config, scene, reduced);
}
