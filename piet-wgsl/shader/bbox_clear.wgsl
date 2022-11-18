// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

#import config

@group(0) @binding(0)
var<storage> config: Config;

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
