// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

#import config
#import bbox
#import clip

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> clip_inp: array<ClipInp>;

@group(0) @binding(2)
var<storage> path_bboxes: array<PathBbox>;

@group(0) @binding(3)
var<storage, read_write> reduced: array<Bic>;

@group(0) @binding(4)
var<storage, read_write> clip_out: array<ClipEl>;

let WG_SIZE = 256u;
var<workgroup> sh_bic: array<Bic, WG_SIZE>;
var<workgroup> sh_parent: array<u32, WG_SIZE>;
var<workgroup> sh_path_ix: array<u32, WG_SIZE>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let inp = clip_inp[global_id.x].path_ix;
    let is_push = inp >= 0;
    var bic = Bic(1u - u32(is_push), u32(is_push));
    // reverse scan of bicyclic semigroup
    sh_bic[local_id.x] = bic;
    for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
        workgroupBarrier();
        if local_id.x + (1u << i) < WG_SIZE {
            let other = sh_bic[local_id.x + (1u << i)];
            bic = bic_combine(bic, other);
        }
        workgroupBarrier();
        sh_bic[local_id.x] = bic;
    }
    if local_id.x == 0u {
        reduced[wg_id.x] = bic;
    }
    workgroupBarrier();
    let size = sh_bic[0].b;
    bic = Bic();
    if is_push && bic.a == 0u {
        let local_ix = size - bic.b - 1u;
        sh_parent[local_ix] = local_id.x;
        sh_path_ix[local_ix] = u32(inp);
    }
    workgroupBarrier();
    // TODO: possibly do forward scan here if depth can exceed wg size
    if local_id.x < size {
        let path_ix = sh_path_ix[local_id.x];
        let path_bbox = path_bboxes[path_ix];
        let parent_ix = sh_parent[local_id.x] + wg_id.x * WG_SIZE;
        let bbox = vec4(f32(path_bbox.x0), f32(path_bbox.y0), f32(path_bbox.x1), f32(path_bbox.y1));
        clip_out[global_id.x] = ClipEl(parent_ix, bbox);
    }
}
