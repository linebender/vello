// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Note: this is the non-atomic version
struct Tile {
    backdrop: i32,
    segments: u32,
}

#import config

@group(0) @binding(0)
var<uniform> config: Config;

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
    if local_id.x < width_in_tiles {
        backdrop = tiles[ix].backdrop;
    }
    sh_backdrop[local_id.x] = backdrop;
    // iterate log2(WG_SIZE) times
    for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
        workgroupBarrier();
        if local_id.x >= (1u << i) {
            backdrop += sh_backdrop[local_id.x - (1u << i)];
        }
        workgroupBarrier();
        sh_backdrop[local_id.x] = backdrop;
    }
    if local_id.x < width_in_tiles {
        tiles[ix].backdrop = backdrop;
    }
}
