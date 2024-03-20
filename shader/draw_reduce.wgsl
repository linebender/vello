// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

#import config
#import drawtag

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> scene: array<u32>;

@group(0) @binding(2)
var<storage, read_write> reduced: array<DrawMonoid>;

const WG_SIZE = 256u;

var<workgroup> sh_scratch: array<DrawMonoid, WG_SIZE>;

#import util

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let num_blocks_total = (config.n_drawobj + (WG_SIZE - 1u)) / WG_SIZE;
    // When the number of blocks exceeds the workgroup size, divide
    // the work evenly so each workgroup handles n_blocks / wg, with
    // the low workgroups doing one more each to handle the remainder.
    let n_blocks_base = num_blocks_total / WG_SIZE;
    let remainder = num_blocks_total % WG_SIZE;
    let first_block = n_blocks_base * wg_id.x + min(wg_id.x, remainder);
    let n_blocks = n_blocks_base + u32(wg_id.x < remainder);
    var block_index = first_block * WG_SIZE + local_id.x;
    var agg = draw_monoid_identity();
    for (var i = 0u; i < n_blocks; i++) {
        let tag_word = read_draw_tag_from_scene(block_index);
        agg = combine_draw_monoid(agg, map_draw_tag(tag_word));
        block_index += WG_SIZE;
    }
    sh_scratch[local_id.x] = agg;
    for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
        workgroupBarrier();
        if local_id.x + (1u << i) < WG_SIZE {
            let other = sh_scratch[local_id.x + (1u << i)];
            agg = combine_draw_monoid(agg, other);
        }
        workgroupBarrier();
        sh_scratch[local_id.x] = agg;
    }
    if local_id.x == 0u {
        reduced[wg_id.x] = agg;
    }
}
