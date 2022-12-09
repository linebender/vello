// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

#import config

@group(0) @binding(0)
var<uniform> config: Config;

struct PathBbox {
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    linewidth: f32,
    trans_ix: u32,
}

@group(0) @binding(1)
var<storage, read_write> path_bboxes: array<PathBbox>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let ix = global_id.x;
    if ix < config.n_path {
        path_bboxes[ix].x0 = 0x7fffffff;
        path_bboxes[ix].y0 = 0x7fffffff;
        path_bboxes[ix].x1 = -0x80000000;
        path_bboxes[ix].y1 = -0x80000000;
    }
}
