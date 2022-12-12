// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// The annotated bounding box for a path. It has been transformed,
// but contains a link to the active transform, mostly for gradients.
// Coordinates are integer pixels (for the convenience of atomic update)
// but will probably become fixed-point fractions for rectangles.
struct PathBbox {
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    linewidth: f32,
    trans_ix: u32,
}

fn bbox_intersect(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4(max(a.xy, b.xy), min(a.zw, b.zw));
}
