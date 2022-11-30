// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Path segment decoding for the full case.

// In the simple case, path segments are decoded as part of the coarse
// path rendering stage. In the full case, they are separated, as the
// decoding process also generates bounding boxes, and those in turn are
// used for tile allocation and clipping; actual coarse path rasterization
// can't proceed until those are complete.

// There's some duplication of the decoding code but we won't worry about
// that just now. Perhaps it could be factored more nicely later.

#import config
#import pathtag
#import cubic

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> scene: array<u32>;

@group(0) @binding(2)
var<storage> tag_monoids: array<TagMonoid>;

struct AtomicPathBbox {
    x0: atomic<i32>,
    y0: atomic<i32>,
    x1: atomic<i32>,
    y1: atomic<i32>,
    linewidth: f32,
    trans_ix: u32,
}

@group(0) @binding(3)
var<storage, read_write> path_bboxes: array<AtomicPathBbox>;


@group(0) @binding(4)
var<storage, read_write> cubics: array<Cubic>;

// Monoid is yagni, for future optimization

// struct BboxMonoid {
//     bbox: vec4<f32>,
//     flags: u32,
// }

// let FLAG_RESET_BBOX = 1u;
// let FLAG_SET_BBOX = 2u;

// fn combine_bbox_monoid(a: BboxMonoid, b: BboxMonoid) -> BboxMonoid {
//     var c: BboxMonoid;
//     c.bbox = b.bbox;
//     // TODO: previous-me thought this should be gated on b & SET_BBOX == false also
//     if (a.flags & FLAG_RESET_BBOX) == 0u && b.bbox.z <= b.bbox.x && b.bbox.w <= b.bbox.y {
//         c.bbox = a.bbox;
//     } else if (a.flags & FLAG_RESET_BBOX) == 0u && (b.flags & FLAG_SET_BBOX) == 0u ||
//         (a.bbox.z > a.bbox.x || a.bbox.w > a.bbox.y)
//     {
//         c.bbox = vec4<f32>(min(a.bbox.xy, c.bbox.xy), max(a.bbox.xw, c.bbox.zw));
//     }
//     c.flags = (a.flags & FLAG_SET_BBOX) | b.flags;
//     c.flags |= (a.flags & FLAG_RESET_BBOX) << 1u;
//     return c;
// }

// fn bbox_monoid_identity() -> BboxMonoid {
//     return BboxMonoid();
// }

var<private> pathdata_base: u32;

fn read_f32_point(ix: u32) -> vec2<f32> {
    let x = bitcast<f32>(scene[pathdata_base + ix]);
    let y = bitcast<f32>(scene[pathdata_base + ix + 1u]);
    return vec2(x, y);
}

fn read_i16_point(ix: u32) -> vec2<f32> {
    let raw = scene[pathdata_base + ix];
    let x = f32(i32(raw << 16u) >> 16u);
    let y = f32(i32(raw) >> 16u);
    return vec2(x, y);
}

struct Transform {
    matrx: vec4<f32>,
    translate: vec2<f32>,
}

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

fn transform_apply(transform: Transform, p: vec2<f32>) -> vec2<f32> {
    return transform.matrx.xy * p.x + transform.matrx.zw * p.y + transform.translate;
}

fn round_down(x: f32) -> i32 {
    return i32(floor(x));
}

fn round_up(x: f32) -> i32 {
    return i32(ceil(x));
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let ix = global_id.x;
    let tag_word = scene[config.pathtag_base + (ix >> 2u)];
    pathdata_base = config.pathdata_base;
    let shift = (ix & 3u) * 8u;
    var tm = reduce_tag(tag_word & ((1u << shift) - 1u));
    tm = combine_tag_monoid(tag_monoids[ix >> 2u], tm);
    var tag_byte = (tag_word >> shift) & 0xffu;

    let out = &path_bboxes[tm.path_ix];
    let linewidth = bitcast<f32>(scene[config.linewidth_base + tm.linewidth_ix]);
    if (tag_byte & PATH_TAG_PATH) != 0u {
        (*out).linewidth = linewidth;
        (*out).trans_ix = tm.trans_ix;
    }
    // Decode path data
    let seg_type = tag_byte & PATH_TAG_SEG_TYPE;
    if seg_type != 0u {
        var p0: vec2<f32>;
        var p1: vec2<f32>;
        var p2: vec2<f32>;
        var p3: vec2<f32>;
        if (tag_byte & PATH_TAG_F32) != 0u {
            p0 = read_f32_point(tm.pathseg_offset);
            p1 = read_f32_point(tm.pathseg_offset + 2u);
            if seg_type >= PATH_TAG_QUADTO {
                p2 = read_f32_point(tm.pathseg_offset + 4u);
                if seg_type == PATH_TAG_CUBICTO {
                    p3 = read_f32_point(tm.pathseg_offset + 6u);
                }
            }
        } else {
            p0 = read_i16_point(tm.pathseg_offset);
            p1 = read_i16_point(tm.pathseg_offset + 1u);
            if seg_type >= PATH_TAG_QUADTO {
                p2 = read_i16_point(tm.pathseg_offset + 2u);
                if seg_type == PATH_TAG_CUBICTO {
                    p3 = read_i16_point(tm.pathseg_offset + 3u);
                }
            }
        }
        let transform = read_transform(config.transform_base, tm.trans_ix);
        p0 = transform_apply(transform, p0);
        p1 = transform_apply(transform, p1);
        var bbox = vec4(min(p0, p1), max(p0, p1));
        // Degree-raise
        if seg_type == PATH_TAG_LINETO {
            p3 = p1;
            p2 = mix(p3, p0, 1.0 / 3.0);
            p1 = mix(p0, p3, 1.0 / 3.0);
        } else if seg_type >= PATH_TAG_QUADTO {
            p2 = transform_apply(transform, p2);
            bbox = vec4(min(bbox.xy, p2), max(bbox.zw, p2));
            if seg_type == PATH_TAG_CUBICTO {
                p3 = transform_apply(transform, p3);
                bbox = vec4(min(bbox.xy, p3), max(bbox.zw, p3));
            } else {
                p3 = p2;
                p2 = mix(p1, p2, 1.0 / 3.0);
                p1 = mix(p1, p0, 1.0 / 3.0);
            }
        }
        var stroke = vec2(0.0, 0.0);
        if linewidth >= 0.0 {
            // See https://www.iquilezles.org/www/articles/ellipses/ellipses.htm
            // This is the correct bounding box, but we're not handling rendering
            // in the isotropic case, so it may mismatch.
            stroke = 0.5 * linewidth * vec2(length(transform.matrx.xz), length(transform.matrx.yw));
            bbox += vec4(-stroke, stroke);
        }
        let flags = u32(linewidth >= 0.0);
        cubics[global_id.x] = Cubic(p0, p1, p2, p3, stroke, tm.path_ix, flags);
        // Update bounding box using atomics only. Computing a monoid is a
        // potential future optimization.
        if bbox.z > bbox.x || bbox.w > bbox.y {
            atomicMin(&(*out).x0, round_down(bbox.x));
            atomicMin(&(*out).y0, round_down(bbox.y));
            atomicMax(&(*out).x1, round_up(bbox.z));
            atomicMax(&(*out).y1, round_up(bbox.w));
        }
    }
}
