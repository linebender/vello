// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

#import config
#import bbox
#import clip
#import drawtag

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> clip_inp: array<ClipInp>;

@group(0) @binding(2)
var<storage> path_bboxes: array<PathBbox>;

@group(0) @binding(3)
var<storage> reduced: array<Bic>;

@group(0) @binding(4)
var<storage> clip_els: array<ClipEl>;

@group(0) @binding(5)
var<storage, read_write> draw_monoids: array<DrawMonoid>;

@group(0) @binding(6)
var<storage, read_write> clip_bboxes: array<vec4<f32>>;

let WG_SIZE = 256u;
var<workgroup> sh_bic: array<Bic, 510 >;
var<workgroup> sh_stack: array<u32, WG_SIZE>;
var<workgroup> sh_stack_bbox: array<vec4<f32>, WG_SIZE>;
var<workgroup> sh_bbox: array<vec4<f32>, WG_SIZE>;
var<workgroup> sh_link: array<i32, WG_SIZE>;

fn search_link(bic: ptr<function, Bic>, ix_in: u32) -> i32 {
    var ix = ix_in;
    var j = 0u;
    while j < firstTrailingBit(WG_SIZE) {
        let base = 2u * WG_SIZE - (2u << (firstTrailingBit(WG_SIZE) - j));
        if ((ix >> j) & 1u) != 0u {
            let test = bic_combine(sh_bic[base + (ix >> j) - 1u], *bic);
            if test.b > 0u {
                break;
            }
            *bic = test;
            ix -= 1u << j;
        }
        j += 1u;
    }
    if ix > 0u {
        while j > 0u {
            j -= 1u;
            let base = 2u * WG_SIZE - (2u << (firstTrailingBit(WG_SIZE) - j));
            let test = bic_combine(sh_bic[base + (ix >> j) - 1u], *bic);
            if test.b == 0u {
                *bic = test;
                ix -= 1u << j;
            }
        }
    }
    if ix > 0u {
        return i32(ix) - 1;
    } else {
        return i32(~0u - (*bic).a);
    }
}

fn load_clip_path(ix: u32) -> i32 {
    if ix < config.n_clip {
        return clip_inp[ix].path_ix;
    } else {
        return -2147483648;
        // literal too large?
        // return 0x80000000;
    }
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    var bic: Bic;
    if local_id.x < wg_id.x {
        bic = reduced[local_id.x];
    }
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
    workgroupBarrier();
    let stack_size = sh_bic[0].b;
    // TODO: if stack depth > WG_SIZE desired, scan here

    // binary search in stack
    let sp = WG_SIZE - 1u - local_id.x;
    var ix = 0u;
    for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
        let probe = ix + ((WG_SIZE / 2u) >> i);
        if sp < sh_bic[probe].b {
            ix = probe;
        }
    }
    let b = sh_bic[ix].b;
    var bbox = vec4(-1e9, -1e9, 1e9, 1e9);
    if sp < b {
        let el = clip_els[ix * WG_SIZE + b - sp - 1u];
        sh_stack[local_id.x] = el.parent_ix;
        bbox = el.bbox;
    }
    // forward scan of bbox values of prefix stack
    for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
        sh_stack_bbox[local_id.x] = bbox;
        workgroupBarrier();
        if local_id.x >= (1u << i) {
            bbox = bbox_intersect(sh_stack_bbox[local_id.x - (1u << i)], bbox);
        }
        workgroupBarrier();
    }
    sh_stack_bbox[local_id.x] = bbox;

    // Read input and compute Bic binary tree
    let inp = load_clip_path(global_id.x);
    let is_push = inp >= 0;
    bic = Bic(1u - u32(is_push), u32(is_push));
    sh_bic[local_id.x] = bic;
    if is_push {
        let path_bbox = path_bboxes[inp];
        bbox = vec4(f32(path_bbox.x0), f32(path_bbox.y0), f32(path_bbox.x1), f32(path_bbox.y1));
    } else {
        bbox = vec4(-1e9, -1e9, 1e9, 1e9);
    }
    var inbase = 0u;
    for (var i = 0u; i < firstTrailingBit(WG_SIZE) - 1u; i += 1u) {
        let outbase = 2u * WG_SIZE - (1u << (firstTrailingBit(WG_SIZE) - i));
        workgroupBarrier();
        if local_id.x < 1u << (firstTrailingBit(WG_SIZE) - 1u - i) {
            let in_off = inbase + local_id.x * 2u;
            sh_bic[outbase + local_id.x] = bic_combine(sh_bic[in_off], sh_bic[in_off + 1u]);
        }
        inbase = outbase;
    }
    workgroupBarrier();
    // search for predecessor node
    bic = Bic();
    var link = search_link(&bic, local_id.x);
    sh_link[local_id.x] = link;
    workgroupBarrier();
    let grandparent = select(link - 1, sh_link[link], link >= 0);
    var parent: i32;
    if link >= 0 {
        parent = i32(wg_id.x * WG_SIZE) + link;
    } else if link + i32(stack_size) >= 0 {
        parent = i32(sh_stack[i32(WG_SIZE) + link]);
    } else {
        parent = -1;
    }
    // bbox scan (intersect) across parent links
    for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
        if i != 0u {
            sh_link[local_id.x] = link;
        }
        sh_bbox[local_id.x] = bbox;
        workgroupBarrier();
        if link >= 0 {
            bbox = bbox_intersect(sh_bbox[link], bbox);
            link = sh_link[link];
        }
        workgroupBarrier();
    }
    if link + i32(stack_size) >= 0 {
        bbox = bbox_intersect(sh_stack_bbox[i32(WG_SIZE) + link], bbox);
    }
    // At this point, bbox is the intersection of bboxes on the path to the root
    sh_bbox[local_id.x] = bbox;
    workgroupBarrier();

    if !is_push && global_id.x < config.n_clip {
        // Fix up drawmonoid so path_ix of EndClip matches BeginClip
        let parent_clip = clip_inp[parent];
        let path_ix = parent_clip.path_ix;
        let parent_ix = parent_clip.ix;
        let ix = ~inp;
        draw_monoids[ix].path_ix = u32(path_ix);
        // Make EndClip point to the same draw data as BeginClip
        draw_monoids[ix].scene_offset = draw_monoids[parent_ix].scene_offset;
        if grandparent >= 0 {
            bbox = sh_bbox[grandparent];
        } else if grandparent + i32(stack_size) >= 0 {
            bbox = sh_stack_bbox[i32(WG_SIZE) + grandparent];
        } else {
            bbox = vec4(-1e9, -1e9, 1e9, 1e9);
        }
    }
    clip_bboxes[global_id.x] = bbox;
}
