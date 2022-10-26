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

// Note: this is the non-atomic version
struct Tile {
    backdrop: i32,
    segments: u32,
}

#import config

@group(0) @binding(0)
var<storage> config: Config;

@group(0) @binding(1)
var<storage, read_write> tiles: array<Tile>;

let WG_SIZE = 64u;

var<workgroup> sh_backdrop: array<i32, WG_SIZE>;

// Each workgroup computes the inclusive prefix sum of the backdrops
// in one row of tiles.
@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let width_in_tiles = config.width_in_tiles;
    let ix = wg_id.x * width_in_tiles + local_id.x;
    var backdrop = 0;
    if (local_id.x < width_in_tiles) {
        backdrop = tiles[ix].backdrop;
    }
    sh_backdrop[local_id.x] = backdrop;
    // iterate log2(WG_SIZE) times
    for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
        workgroupBarrier();
        if (local_id.x >= (1u << i)) {
            backdrop += sh_backdrop[local_id.x - (1u << i)];
        }
        workgroupBarrier();
        sh_backdrop[local_id.x] = backdrop;
    }
    if (local_id.x < width_in_tiles) {
        tiles[ix].backdrop = backdrop;
    }
}
