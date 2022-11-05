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

@group(0) @binding(0)
var<storage, read_write> bump: atomic<u32>;

@group(0) @binding(1)
var<storage, read_write> paths: array<u32>;

let WG_SIZE = 256u;

var<workgroup> sh_tile_offset: u32;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {

    let drawobj_ix = global_id.x;
    let tile_count_in = local_id.x + 1u;
    if local_id.x == WG_SIZE - 1u {
        sh_tile_offset = atomicAdd(&bump, tile_count_in);
    }
    workgroupBarrier();
    let tile_offset = sh_tile_offset;
    if drawobj_ix < 3u {
        paths[drawobj_ix] = tile_offset;
    }

}
