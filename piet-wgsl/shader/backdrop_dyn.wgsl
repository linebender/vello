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

// Prefix sum for dynamically allocated backdrops

#import config
#import tile

@group(0) @binding(0)
var<storage> config: Config;

@group(0) @binding(1)
var<storage> paths: array<Path>;

@group(0) @binding(2)
var<storage, read_write> tiles: array<Tile>;

let WG_SIZE = 256u;

var<workgroup> sh_row_width: array<u32, WG_SIZE>;
var<workgroup> sh_row_count: array<u32, WG_SIZE>;
var<workgroup> sh_offset: array<u32, WG_SIZE>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let drawobj_ix = global_id.x;
    var row_count = 0u;
    if drawobj_ix < config.n_drawobj {
        // TODO: when rectangles, path and draw obj are not the same
        let path = paths[drawobj_ix];
        sh_row_width[local_id.x] = path.bbox.z - path.bbox.x;
        row_count = path.bbox.w - path.bbox.y;
        sh_offset[local_id.x] = path.tiles;
    }
    sh_row_count[local_id.x] = row_count;

    // Prefix sum of row counts
    for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
        workgroupBarrier();
        if local_id.x >= (1u << i) {
            row_count += sh_row_count[local_id.x - (1u << i)];
        }
        workgroupBarrier();
        sh_row_count[local_id.x] = row_count;
    }
    workgroupBarrier();
    let total_rows = sh_row_count[WG_SIZE - 1u];
    for (var row = local_id.x; row < total_rows; row += WG_SIZE) {
        var el_ix = 0u;
        for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
            let probe = el_ix + ((WG_SIZE / 2u) >> i);
            if row >= sh_row_count[probe - 1u] {
                el_ix = probe;
            }
        }
        let width = sh_row_width[el_ix];
        if width > 0u {
            var seq_ix = row - select(0u, sh_row_count[el_ix - 1u], el_ix > 0u);
            var tile_ix = sh_offset[el_ix] + seq_ix * width;
            var sum = tiles[tile_ix].backdrop;
            for (var x = 1u; x < width; x += 1u) {
                tile_ix += 1u;
                sum += tiles[tile_ix].backdrop;
                tiles[tile_ix].backdrop = sum;
            }
        }
    }
}
