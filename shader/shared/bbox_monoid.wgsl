// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

struct BboxMonoid {
    bbox: vec4<f32>,
    flags: u32,
}

let FLAG_RESET_BBOX = 1u;
let FLAG_SET_BBOX = 2u;

// Technically this is a semigroup with a left identity rather than a
// true monoid, but that is good enough for our purposes.
fn combine_bbox_monoid(a: BboxMonoid, b: BboxMonoid) -> BboxMonoid {
    var bbox = b.bbox;
    if (b.flags & FLAG_SET_BBOX) == 0u && (a.flags & FLAG_RESET_BBOX) == 0u {
        if bbox.z <= bbox.x && bbox.w <= bbox.y {
            bbox = a.bbox;
        } else if a.bbox.z > a.bbox.x || a.bbox.w > a.bbox.y {
            bbox = vec4(min(a.bbox.xy, bbox.xy), max(a.bbox.zw, bbox.zw));
        }
    }
    let flags = ((a.flags | (a.flags << 1u)) & FLAG_SET_BBOX) | b.flags;
    return BboxMonoid(bbox, flags);
}

