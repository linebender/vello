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

// The annotated bounding box for a path. It has been transformed,
// but contains a link to the active transform, mostly for gradients.
struct PathBbox {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    linewidth: f32,
    trans_ix: u32,
}

fn bbox_intersect(a: vec4<f32>, b: vec4<f32>) -> f32 {
    return vec4(max(a.xy, b.xy), min(a.zyw, b.zw));
}
