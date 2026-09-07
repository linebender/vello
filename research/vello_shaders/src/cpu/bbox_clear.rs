// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

use vello_encoding::{ConfigUniform, PathBbox};

use super::CpuBinding;

fn bbox_clear_main(config: &ConfigUniform, path_bboxes: &mut [PathBbox]) {
    for i in 0..(config.layout.n_paths as usize) {
        path_bboxes[i].x0 = 0x7fff_ffff;
        path_bboxes[i].y0 = 0x7fff_ffff;
        path_bboxes[i].x1 = -0x8000_0000;
        path_bboxes[i].y1 = -0x8000_0000;
    }
}

pub fn bbox_clear(_n_wg: u32, resources: &[CpuBinding<'_>]) {
    let config = resources[0].as_typed();
    let mut path_bboxes = resources[1].as_slice_mut();
    bbox_clear_main(&config, &mut path_bboxes);
}
