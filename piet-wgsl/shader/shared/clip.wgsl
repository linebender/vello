// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense
 
struct Bic {
    a: u32,
    b: u32,
}

fn bic_combine(x: Bic, y: Bic) -> Bic {
    let m = min(x.b, y.a);
    return Bic(x.a + y.a - m, x.b + y.b - m);
}

struct ClipEl {
    parent_ix: u32,
    bbox: vec4<f32>,
}
