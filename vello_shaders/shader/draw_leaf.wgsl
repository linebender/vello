// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Finish prefix sum of drawtags, decode draw objects.

#import config
#import clip
#import drawtag
#import bbox
#import transform

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> scene: array<u32>;

@group(0) @binding(2)
var<storage> reduced: array<DrawMonoid>;

@group(0) @binding(3)
var<storage> path_bbox: array<PathBbox>;

@group(0) @binding(4)
var<storage, read_write> draw_monoid: array<DrawMonoid>;

@group(0) @binding(5)
var<storage, read_write> info: array<u32>;

@group(0) @binding(6)
var<storage, read_write> clip_inp: array<ClipInp>;

#import util

const WG_SIZE = 256u;

fn read_transform(transform_base: u32, ix: u32) -> Transform {
    let base = transform_base + ix * 6u;
    let c0 = bitcast<f32>(scene[base]);
    let c1 = bitcast<f32>(scene[base + 1u]);
    let c2 = bitcast<f32>(scene[base + 2u]);
    let c3 = bitcast<f32>(scene[base + 3u]);
    let c4 = bitcast<f32>(scene[base + 4u]);
    let c5 = bitcast<f32>(scene[base + 5u]);
    let matrx = vec4(c0, c1, c2, c3);
    let translate = vec2(c4, c5);
    return Transform(matrx, translate);
}

var<workgroup> sh_scratch: array<DrawMonoid, WG_SIZE>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    // Reduce prefix of workgroups up to this one
    var agg = draw_monoid_identity();
    if local_id.x < wg_id.x {
        agg = reduced[local_id.x];
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
    // Two barriers can be eliminated if we use separate shared arrays
    // for prefix and intra-workgroup prefix sum.
    workgroupBarrier();
    var prefix = sh_scratch[0];

    // This is the same division of work as draw_reduce.
    let num_blocks_total = (config.n_drawobj + WG_SIZE - 1u) / WG_SIZE;
    let n_blocks_base = num_blocks_total / WG_SIZE;
    let remainder = num_blocks_total % WG_SIZE;
    let first_block = n_blocks_base * wg_id.x + min(wg_id.x, remainder);
    let n_blocks = n_blocks_base + u32(wg_id.x < remainder);
    var block_start = first_block * WG_SIZE;
    let blocks_end = block_start + n_blocks * WG_SIZE;
    while block_start != blocks_end {
        let ix = block_start + local_id.x;
        let tag_word = read_draw_tag_from_scene(ix);
        agg = map_draw_tag(tag_word);
        workgroupBarrier();
        sh_scratch[local_id.x] = agg;
        for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
            workgroupBarrier();
            if local_id.x >= 1u << i {
                let other = sh_scratch[local_id.x - (1u << i)];
                agg = combine_draw_monoid(agg, other);
            }
            workgroupBarrier();
            sh_scratch[local_id.x] = agg;
        }
        var m = prefix;
        workgroupBarrier();
        if local_id.x > 0u {
            m = combine_draw_monoid(m, sh_scratch[local_id.x - 1u]);
        }
        // m now contains exclusive prefix sum of draw monoid
        if ix < config.n_drawobj {
            draw_monoid[ix] = m;
        }
        let dd = config.drawdata_base + m.scene_offset;
        let di = m.info_offset;
        if tag_word == DRAWTAG_FILL_COLOR || tag_word == DRAWTAG_FILL_LIN_GRADIENT ||
            tag_word == DRAWTAG_FILL_RAD_GRADIENT || tag_word == DRAWTAG_FILL_SWEEP_GRADIENT ||
            tag_word == DRAWTAG_FILL_IMAGE || tag_word == DRAWTAG_BEGIN_CLIP || tag_word == DRAWTAG_BLURRED_ROUNDED_RECT
        {
            let bbox = path_bbox[m.path_ix];
            // TODO: bbox is mostly yagni here, sort that out. Maybe clips?
            // let x0 = f32(bbox.x0);
            // let y0 = f32(bbox.y0);
            // let x1 = f32(bbox.x1);
            // let y1 = f32(bbox.y1);
            // let bbox_f = vec4(x0, y0, x1, y1);
            var transform = Transform();
            let draw_flags = bbox.draw_flags;
            if tag_word == DRAWTAG_FILL_LIN_GRADIENT || tag_word == DRAWTAG_FILL_RAD_GRADIENT ||
                tag_word == DRAWTAG_FILL_SWEEP_GRADIENT || tag_word == DRAWTAG_FILL_IMAGE || 
                tag_word == DRAWTAG_BLURRED_ROUNDED_RECT
            {
                transform = read_transform(config.transform_base, bbox.trans_ix);
            }
            switch tag_word {
                case DRAWTAG_FILL_COLOR: {
                    info[di] = draw_flags;
                }
                case DRAWTAG_BEGIN_CLIP: {
                    info[di] = draw_flags;
                }
                case DRAWTAG_FILL_LIN_GRADIENT: {
                    info[di] = draw_flags;
                    var p0 = bitcast<vec2<f32>>(vec2(scene[dd + 1u], scene[dd + 2u]));
                    var p1 = bitcast<vec2<f32>>(vec2(scene[dd + 3u], scene[dd + 4u]));
                    p0 = transform_apply(transform, p0);
                    p1 = transform_apply(transform, p1);
                    let dxy = p1 - p0;
                    let scale = 1.0 / dot(dxy, dxy);
                    let line_xy = dxy * scale;
                    let line_c = -dot(p0, line_xy);
                    info[di + 1u] = bitcast<u32>(line_xy.x);
                    info[di + 2u] = bitcast<u32>(line_xy.y);
                    info[di + 3u] = bitcast<u32>(line_c);
                }
                case DRAWTAG_FILL_RAD_GRADIENT: {
                    // Two-point conical gradient implementation based
                    // on the algorithm at <https://skia.org/docs/dev/design/conical/>
                    // This epsilon matches what Skia uses
                    let GRADIENT_EPSILON = 1.0 / f32(1u << 12u);
                    info[di] = draw_flags;
                    var p0 = bitcast<vec2<f32>>(vec2(scene[dd + 1u], scene[dd + 2u]));
                    var p1 = bitcast<vec2<f32>>(vec2(scene[dd + 3u], scene[dd + 4u]));
                    var r0 = bitcast<f32>(scene[dd + 5u]);
                    var r1 = bitcast<f32>(scene[dd + 6u]);
                    let user_to_gradient = transform_inverse(transform);
                    // Output variables
                    var xform = Transform();
                    var focal_x = 0.0;
                    var radius = 0.0;
                    var kind = 0u;
                    var flags = 0u;
                    if abs(r0 - r1) <= GRADIENT_EPSILON {
                        // When the radii are the same, emit a strip gradient
                        kind = RAD_GRAD_KIND_STRIP;
                        let scaled = r0 / distance(p0, p1);
                        xform = transform_mul(
                            two_point_to_unit_line(p0, p1),
                            user_to_gradient
                        );
                        radius = scaled * scaled;
                    } else {
                        // Assume a two point conical gradient unless the centers
                        // are equal.
                        kind = RAD_GRAD_KIND_CONE;
                        if all(p0 == p1) {
                            kind = RAD_GRAD_KIND_CIRCULAR;
                            // Nudge p0 a bit to avoid denormals.
                            p0 += GRADIENT_EPSILON;
                        }
                        if r1 == 0.0 {
                            // If r1 == 0.0, swap the points and radii
                            flags |= RAD_GRAD_SWAPPED;
                            let tmp_p = p0;
                            p0 = p1;
                            p1 = tmp_p;
                            let tmp_r = r0;
                            r0 = r1;
                            r1 = tmp_r;
                        }
                        focal_x = r0 / (r0 - r1);
                        let cf = (1.0 - focal_x) * p0 + focal_x * p1;
                        radius = r1 / (distance(cf, p1));
                        let user_to_unit_line = transform_mul(
                            two_point_to_unit_line(cf, p1),
                            user_to_gradient
                        );
                        var user_to_scaled = user_to_unit_line;
                        // When r == 1.0, focal point is on circle
                        if abs(radius - 1.0) <= GRADIENT_EPSILON {
                            kind = RAD_GRAD_KIND_FOCAL_ON_CIRCLE;
                            let scale = 0.5 * abs(1.0 - focal_x);
                            user_to_scaled = transform_mul(
                                Transform(vec4(scale, 0.0, 0.0, scale), vec2(0.0)),
                                user_to_unit_line
                            );
                        } else {
                            let a = radius * radius - 1.0;
                            let scale_ratio = abs(1.0 - focal_x) / a;
                            let scale_x = radius * scale_ratio;
                            let scale_y = sqrt(abs(a)) * scale_ratio;
                            user_to_scaled = transform_mul(
                                Transform(vec4(scale_x, 0.0, 0.0, scale_y), vec2(0.0)),
                                user_to_unit_line
                            );
                        }
                        xform = user_to_scaled;
                    }
                    info[di + 1u] = bitcast<u32>(xform.matrx.x);
                    info[di + 2u] = bitcast<u32>(xform.matrx.y);
                    info[di + 3u] = bitcast<u32>(xform.matrx.z);
                    info[di + 4u] = bitcast<u32>(xform.matrx.w);
                    info[di + 5u] = bitcast<u32>(xform.translate.x);
                    info[di + 6u] = bitcast<u32>(xform.translate.y);
                    info[di + 7u] = bitcast<u32>(focal_x);
                    info[di + 8u] = bitcast<u32>(radius);
                    info[di + 9u] = bitcast<u32>((flags << 3u) | kind);
                }
                case DRAWTAG_FILL_SWEEP_GRADIENT: {
                    info[di] = draw_flags;
                    let p0 = bitcast<vec2<f32>>(vec2(scene[dd + 1u], scene[dd + 2u]));
                    let xform = transform_mul(transform, Transform(vec4(1.0, 0.0, 0.0, 1.0), p0));
                    let inv = transform_inverse(xform);
                    info[di + 1u] = bitcast<u32>(inv.matrx.x);
                    info[di + 2u] = bitcast<u32>(inv.matrx.y);
                    info[di + 3u] = bitcast<u32>(inv.matrx.z);
                    info[di + 4u] = bitcast<u32>(inv.matrx.w);
                    info[di + 5u] = bitcast<u32>(inv.translate.x);
                    info[di + 6u] = bitcast<u32>(inv.translate.y);
                    info[di + 7u] = scene[dd + 3u];
                    info[di + 8u] = scene[dd + 4u];
                }
                case DRAWTAG_FILL_IMAGE: {
                    info[di] = draw_flags;
                    let inv = transform_inverse(transform);
                    info[di + 1u] = bitcast<u32>(inv.matrx.x);
                    info[di + 2u] = bitcast<u32>(inv.matrx.y);
                    info[di + 3u] = bitcast<u32>(inv.matrx.z);
                    info[di + 4u] = bitcast<u32>(inv.matrx.w);
                    info[di + 5u] = bitcast<u32>(inv.translate.x);
                    info[di + 6u] = bitcast<u32>(inv.translate.y);
                    info[di + 7u] = scene[dd];
                    info[di + 8u] = scene[dd + 1u];
                    info[di + 9u] = scene[dd + 2u];
                }
                case DRAWTAG_BLURRED_ROUNDED_RECT: {
                    info[di] = draw_flags;
                    let inv = transform_inverse(transform);
                    info[di + 1u] = bitcast<u32>(inv.matrx.x);
                    info[di + 2u] = bitcast<u32>(inv.matrx.y);
                    info[di + 3u] = bitcast<u32>(inv.matrx.z);
                    info[di + 4u] = bitcast<u32>(inv.matrx.w);
                    info[di + 5u] = bitcast<u32>(inv.translate.x);
                    info[di + 6u] = bitcast<u32>(inv.translate.y);
                    info[di + 7u] = scene[dd + 1u];
                    info[di + 8u] = scene[dd + 2u];
                    info[di + 9u] = scene[dd + 3u];
                    info[di + 10u] = scene[dd + 4u];
                }
                default: {}
            }
        }
        if tag_word == DRAWTAG_BEGIN_CLIP || tag_word == DRAWTAG_END_CLIP {
            var path_ix = ~ix;
            if tag_word == DRAWTAG_BEGIN_CLIP {
                path_ix = m.path_ix;
            }
            clip_inp[m.clip_ix] = ClipInp(ix, i32(path_ix));
        }
        block_start += WG_SIZE;
        // break here on end to save monoid aggregation?
        prefix = combine_draw_monoid(prefix, sh_scratch[WG_SIZE - 1u]);
    }
}

fn two_point_to_unit_line(p0: vec2<f32>, p1: vec2<f32>) -> Transform {
    let tmp1 = from_poly2(p0, p1);
    let inv = transform_inverse(tmp1);
    let tmp2 = from_poly2(vec2(0.0), vec2(1.0, 0.0));
    return transform_mul(tmp2, inv);
}

fn from_poly2(p0: vec2<f32>, p1: vec2<f32>) -> Transform {
    return Transform(
        vec4(p1.y - p0.y, p0.x - p1.x, p1.x - p0.x, p1.y - p0.y),
        vec2(p0.x, p0.y)
    );
}
