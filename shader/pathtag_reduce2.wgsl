// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// This shader is the second stage of reduction for the pathtag
// monoid scan, needed when the number of tags is large.

#import config
#import pathtag

@group(0) @binding(0)
var<storage> reduced_in: array<TagMonoid>;

@group(0) @binding(1)
var<storage, read_write> reduced: array<TagMonoid>;

let LG_WG_SIZE = 8u;
let WG_SIZE = 256u;

var<workgroup> sh_scratch: array<TagMonoid, WG_SIZE>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let ix = global_id.x;
    var agg = reduced_in[ix];
    sh_scratch[local_id.x] = agg;
    for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
        workgroupBarrier();
        if local_id.x + (1u << i) < WG_SIZE {
            let other = sh_scratch[local_id.x + (1u << i)];
            agg = combine_tag_monoid(agg, other);
        }
        workgroupBarrier();
        sh_scratch[local_id.x] = agg;
    }
    if local_id.x == 0u {
        reduced[ix >> LG_WG_SIZE] = agg;
    }
}
