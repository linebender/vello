// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

struct Segment {
    origin: vec2<f32>,
    delta: vec2<f32>,
    y_edge: f32,
    next: u32,
}
