// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_encoding::{ConfigUniform, PathBbox};

use crate::cpu_dispatch::CpuBinding;

fn bbox_clear_main(config: &ConfigUniform, path_bboxes: &mut [PathBbox]) {
    for i in 0..(config.layout.n_paths as usize) {
        path_bboxes[i].x0 = 0x7fff_ffff;
        path_bboxes[i].y0 = 0x7fff_ffff;
        path_bboxes[i].x1 = -0x8000_0000;
        path_bboxes[i].y1 = -0x8000_0000;
    }
}

pub fn bbox_clear(_n_wg: u32, resources: &[CpuBinding]) {
    let r0 = resources[0].as_buf();
    let mut r1 = resources[1].as_buf();
    let config = bytemuck::from_bytes(&r0);
    let path_bboxes = bytemuck::cast_slice_mut(r1.as_mut());
    bbox_clear_main(config, path_bboxes);
}
