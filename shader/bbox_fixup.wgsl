// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

#import config
#import pathtag
#import bbox
#import bbox_monoid

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> tag_monoids: array<TagMonoid>;

@group(0) @binding(2)
var<storage> bbox_reduced: array<BboxMonoid>;

@group(0) @binding(3)
var<storage, read_write> path_bboxes: array<PathBbox>;

let WG_SIZE = 256u;
var<workgroup> sh_bbox: array<BboxMonoid, WG_SIZE>;

fn round_down(x: f32) -> i32 {
    return i32(floor(x));
}

fn round_up(x: f32) -> i32 {
    return i32(ceil(x));
}

// In the configuration with <= 64k pathtags, there's only one
// workgroup here, so the distinction between global and local is
// not meaningful. But we'll probably want to #ifdef a larger
// configuration, in which we also bind a doubly reduced buffer.
@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    var agg: BboxMonoid;
    if global_id.x * WG_SIZE < config.n_pathtag {
        agg = bbox_reduced[global_id.x];
    }
    sh_bbox[local_id.x] = agg;
    for (var i = 0u; i < firstTrailingBit(WG_SIZE); i++) {
        workgroupBarrier();
        if local_id.x >= 1u << i {
            let other = sh_bbox[local_id.x - (1u << i)];
            agg = combine_bbox_monoid(other, agg);
        }
        workgroupBarrier();
        sh_bbox[local_id.x] = agg;
    }
    // Explanation of this trick: we don't need to fix up first bbox.
    // By offsetting the index, we can use the inclusive scan.
    let ix = global_id.x + 1u;
    if ix * WG_SIZE < config.n_pathtag {
        // First path of the workgroup.
        let path_ix = tag_monoids[ix * (WG_SIZE / 4u)].path_ix;
        if (agg.flags & FLAG_RESET_BBOX) == 0u && (agg.bbox.z > agg.bbox.x || agg.bbox.w > agg.bbox.y) {
            let out = &path_bboxes[path_ix];
            // TODO: casting goes away
            var bbox = vec4(f32((*out).x0), f32((*out).y0), f32((*out).x1), f32((*out).y1));
            if bbox.z > bbox.x || bbox.w > bbox.y {
                bbox = vec4(min(agg.bbox.xy, bbox.xy), max(agg.bbox.zw, bbox.zw));
            } else {
                bbox = agg.bbox;
            }
            (*out).x0 = round_down(bbox.x);
            (*out).y0 = round_down(bbox.y);
            (*out).x1 = round_up(bbox.z);
            (*out).y1 = round_up(bbox.w);
        }
    }
}
