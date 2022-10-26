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

// Finish prefix sum of drawtags, decode draw objects.

#import config
#import drawtag
#import bbox

@group(0) @binding(0)
var<storage> config: Config;

@group(0) @binding(1)
var<storage> scene: array<u32>;

@group(0) @binding(2)
var<storage> reduced: array<DrawMonoid>;

@group(0) @binding(3)
var<storage, read_write> draw_monoid: array<DrawMonoid>;

@group(0) @binding(4)
var<storage> path_bbox: array<PathBbox>;

@group(0) @binding(5)
var<storage, read_write> info: array<u32>;

let WG_SIZE = 256;

var<workgroup> sh_scratch: array<DrawMonoid, WG_SIZE>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let ix = global_id.x;
    let tag_word = scene[config.drawtag_base + ix];
    let agg = map_draw_tag(tag_word);
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
    workgroupBarrier();
    var m = draw_monoid_identity();
    if wg_id.x > 0u {
        m = parent[wg_id.x - 1u];
    }
    if local_id.x > 0u {
        m = combine_draw_monoid(m, sh_scratch[local_id.x - 1u]);
    }
    // m now contains exclusive prefix sum of draw monoid
    draw_monoid[ix] = m;
    let dd = config.drawdata_base + m.scene_offset;
    let di = m.info_offset;
    if tag_word == DRAWTAG_FILL_COLOR || tag_word == DRAWTAG_FILL_LIN_GRADIENT ||
        tag_word == DRAWTAG_FILL_RAD_GRADIENT || tag_word == DRAWTAG_FILL_IMAGE ||
        tag_word == DRAWTAG_BEGIN_CLIP
    {
        let bbox = path_bbox[m.path_ix];
        let x0 = f32(bbox.x0) - 32768.0;
        let y0 = f32(bbox.y0) - 32768.0;
        let x1 = f32(bbox.x1) - 32768.0;
        let y1 = f32(bbox.y1) - 32768.0;
        let bbox_f = vec4(x0, y0, x1, y1);
        let fill_mode = u32(bbox.linewidth >= 0.0);
        var mat: vec4<f32>;
        var translate: vec2<f32>;
        var linewidth = bbox.linewidth;
        if linewidth >= 0.0 || tag_word == DRAWTAG_FILL_LIN_GRADIENT || tag_word == DRAWTAG_FILL_RAD_GRADIENT {
            // TODO: retrieve transform from scene. Packed?
        }
        if linewidth >= 0.0 {
            // Note: doesn't deal with anisotropic case
            linewidth *= sqrt(abs(mat.x * mat.w - mat.y * mat.z)); 
        }
        switch tag_word {
            case DRAWTAG_FILL_COLOR, DRAWTAG_FILL_IMAGE: {
                info[di] = bitcast<u32>(linewidth);
            }
            case DRAWTAG_FILL_LIN_GRADIENT: {
                info[di] = bitcast<u32>(linewidth);
                var p0 = bitcast<vec2<f32>>(vec2(scene[dd + 1u], scene[dd + 2u]));
                var p1 = bitcast<vec2<f32>>(vec2(scene[dd + 3u], scene[dd + 4u]));
                p0 = mat.xy * p0.x + mat.zw * p0.y + translate;
                p1 = mat.xy * p1.x + mat.zw * p1.y + translate;
                let dxy = p1 - p0;
                let scale = 1.0 / dot(dxy, dxy);
                let line_xy = dxy * scale;
                let line_c = -dot(p0, line_xy);
                info[di + 1u] = bitcast<u32>(line_xy.x);
                info[di + 2u] = bitcast<u32>(line_xy.y);
                info[di + 3u] = bitcast<u32>(line_c);
            }
            case DRAWTAG_FILL_RAD_GRADIENT: {
                info[di] = bitcast<u32>(linewidth);
                var p0 = bitcast<vec2<f32>>(vec2(scene[dd + 1u], scene[dd + 2u]));
                var p1 = bitcast<vec2<f32>>(vec2(scene[dd + 3u], scene[dd + 4u]));
                let r0 = bitcast<f32>(scene[dd + 5u]);
                let r1 = bitcast<f32>(scene[dd + 6u]);
                let inv_det = 1.0 / (mat.x * mat.w - mat.y * mat.z);
                let inv_mat = inv_det * vec4(mat.w, -mat.y, -mat.z, mat.x);
                var inv_tr = inv_mat.xz * translate.x + inv_mat.yw * translate.y;
                inv_tr += p0;
                let center1 = p1 - p0;
                let rr = r1 / (r1 - r0);
                let ra_inv = rr / (r1 * r1 - dot(center1, center1));
                let c1 = center1 * ra_inv;
                let ra = rr * ra_inv;
                let roff = rr - 1.0;
                info[di + 1u] = bitcast<u32>(inv_mat.x);
                info[di + 2u] = bitcast<u32>(inv_mat.y);
                info[di + 3u] = bitcast<u32>(inv_mat.z);
                info[di + 4u] = bitcast<u32>(inv_mat.w);
                info[di + 5u] = bitcast<u32>(inv_tr.x);
                info[di + 6u] = bitcast<u32>(inv_tr.y);
                info[di + 7u] = bitcast<u32>(c1.x);
                info[di + 8u] = bitcast<u32>(c1.y);
                info[di + 9u] = bitcast<u32>(ra);
                info[di + 10u] = bitcast<u32>(roff);
            }
            default: {}
        }
    }
}