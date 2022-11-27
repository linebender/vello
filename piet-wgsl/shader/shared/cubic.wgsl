// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

struct Cubic {
    p0: vec2<f32>,
    p1: vec2<f32>,
    p2: vec2<f32>,
    p3: vec2<f32>,
    stroke: vec2<f32>,
    path_ix: u32,
    flags: u32,
}

let CUBIC_IS_STROKE = 1u;
