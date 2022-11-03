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
#import pathtag

@group(0) @binding(0)
var<storage> config: Config;

@group(0) @binding(1)
var<storage> scene: array<u32>;

@group(0) @binding(2)
var<storage> reduced: array<TagMonoid>;

@group(0) @binding(3)
var<storage, read_write> tag_monoids: array<TagMonoid>;

let LG_WG_SIZE = 8u;
let WG_SIZE = 256u;

var<workgroup> sh_parent: array<TagMonoid, WG_SIZE>;
// These could be combined?
var<workgroup> sh_monoid: array<TagMonoid, WG_SIZE>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    var agg = tag_monoid_identity();
    if (local_id.x < wg_id.x) {
        agg = reduced[local_id.x];
    }
    sh_parent[local_id.x] = agg;
    for (var i = 0u; i < LG_WG_SIZE; i += 1u) {
        workgroupBarrier();
        if (local_id.x + (1u << i) < WG_SIZE) {
            let other = sh_parent[local_id.x + (1u << i)];
            agg = combine_tag_monoid(agg, other);
        }
        workgroupBarrier();
        sh_parent[local_id.x] = agg;
    }

    let ix = global_id.x;
    let tag_word = scene[config.pathtag_base + ix];
    agg = reduce_tag(tag_word);
    sh_monoid[local_id.x] = agg;
    for (var i = 0u; i < LG_WG_SIZE; i += 1u) {
        workgroupBarrier();
        if (local_id.x >= 1u << i) {
            let other = sh_monoid[local_id.x - (1u << i)];
            agg = combine_tag_monoid(other, agg);
        }
        workgroupBarrier();
        sh_monoid[local_id.x] = agg;
    }
    // prefix up to this workgroup
    var tm = sh_parent[0];
    if (local_id.x > 0u) {
        tm = combine_tag_monoid(tm, sh_monoid[local_id.x - 1u]);
    }
    // exclusive prefix sum, granularity of 4 tag bytes
    tag_monoids[ix] = tm;
}
