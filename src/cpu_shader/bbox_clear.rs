// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_encoding::{PathBbox, RenderConfig};

pub fn bbox_clear(config: &RenderConfig, path_bboxes: &mut [PathBbox]) {
    for i in 0..(config.gpu.layout.n_paths as usize) {
        path_bboxes[i].x0 = 0x7fff_ffff;
        path_bboxes[i].y0 = 0x7fff_ffff;
        path_bboxes[i].x1 = -0x8000_0000;
        path_bboxes[i].y1 = -0x8000_0000;
    }
}
