// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared encoded blurred rounded rectangle accessors and evaluation.

/// 2x2 linear part of the affine transform (columns [a,b] and [c,d]).
fn get_blurred_rounded_rect_transform(texel0: vec4<u32>) -> mat2x2<f32> {
    return mat2x2<f32>(
        vec2<f32>(bitcast<f32>(texel0.x), bitcast<f32>(texel0.y)),
        vec2<f32>(bitcast<f32>(texel0.z), bitcast<f32>(texel0.w))
    );
}

/// Translation part of the affine transform [tx, ty].
fn get_blurred_rounded_rect_translate(texel1: vec4<u32>) -> vec2<f32> {
    return vec2<f32>(bitcast<f32>(texel1.x), bitcast<f32>(texel1.y));
}

/// Premultiplied rectangle color.
fn get_blurred_rounded_rect_color(texel1: vec4<u32>) -> vec4<f32> { return unpack4x8unorm(texel1.z); }

/// Whether to paint the inverse (`1 - alpha`) of the blur coverage.
fn get_blurred_rounded_rect_invert(texel1: vec4<u32>) -> u32 { return texel1.w; }

fn get_blurred_rounded_rect_exponent(texel2: vec4<u32>) -> f32 { return bitcast<f32>(texel2.x); }

fn get_blurred_rounded_rect_recip_exponent(texel2: vec4<u32>) -> f32 { return bitcast<f32>(texel2.y); }

fn get_blurred_rounded_rect_scale(texel2: vec4<u32>) -> f32 { return bitcast<f32>(texel2.z); }

fn get_blurred_rounded_rect_std_dev_inv(texel2: vec4<u32>) -> f32 { return bitcast<f32>(texel2.w); }

fn get_blurred_rounded_rect_min_edge(texel3: vec4<u32>) -> f32 { return bitcast<f32>(texel3.x); }

fn get_blurred_rounded_rect_w(texel3: vec4<u32>) -> f32 { return bitcast<f32>(texel3.y); }

fn get_blurred_rounded_rect_h(texel3: vec4<u32>) -> f32 { return bitcast<f32>(texel3.z); }

fn get_blurred_rounded_rect_r1(texel3: vec4<u32>) -> f32 { return bitcast<f32>(texel3.w); }

fn get_blurred_rounded_rect_width(texel4: vec4<u32>) -> f32 { return bitcast<f32>(texel4.x); }

fn get_blurred_rounded_rect_height(texel4: vec4<u32>) -> f32 { return bitcast<f32>(texel4.y); }

// Approximation to erf used by Vello Classic and vello_cpu.
fn erf7(x: f32) -> f32 {
    let y = clamp(x * 1.1283791671, -100.0, 100.0);
    let yy = y * y;
    let z = y + (0.24295 + (0.03395 + 0.0104 * yy) * yy) * (y * yy);
    return z / sqrt(1.0 + z * z);
}

// Approximation for the convolution of a gaussian filter with a rounded rectangle, modelled
// after vello_cpu's blurred rounded rectangle painter rather than the Vello Classic shader.
fn calculate_blurred_rounded_rect(
    fragment_pos: vec2<f32>,
    texel0: vec4<u32>,
    texel1: vec4<u32>,
    texel2: vec4<u32>,
    texel3: vec4<u32>,
    texel4: vec4<u32>,
) -> vec4<f32> {
    let transform = get_blurred_rounded_rect_transform(texel0);
    let translate = get_blurred_rounded_rect_translate(texel1);
    let color = get_blurred_rounded_rect_color(texel1);
    let invert = get_blurred_rounded_rect_invert(texel1);
    let exponent = get_blurred_rounded_rect_exponent(texel2);
    let recip_exponent = get_blurred_rounded_rect_recip_exponent(texel2);
    let scale = get_blurred_rounded_rect_scale(texel2);
    let std_dev_inv = get_blurred_rounded_rect_std_dev_inv(texel2);
    let min_edge = get_blurred_rounded_rect_min_edge(texel3);
    let w = get_blurred_rounded_rect_w(texel3);
    let h = get_blurred_rounded_rect_h(texel3);
    let r1 = get_blurred_rounded_rect_r1(texel3);
    let width = get_blurred_rounded_rect_width(texel4);
    let height = get_blurred_rounded_rect_height(texel4);

    let local_xy = transform * fragment_pos + translate;
    // The 0.5 and 0.0 constants correspond to vello_cpu's v1 and v0 respectively.
    let y = local_xy.y - 0.5 * height;
    let y0 = r1 + abs(y) - 0.5 * h;
    let y1 = max(y0, 0.0);

    let x = local_xy.x - 0.5 * width;
    let x0 = r1 + abs(x) - 0.5 * w;
    let x1 = max(x0, 0.0);

    let d_pos = pow(
        pow(x1, exponent) + pow(y1, exponent),
        recip_exponent,
    );
    let d_neg = min(max(x0, y0), 0.0);
    let d = d_pos + d_neg - r1;
    let blur_coverage = scale * (
        erf7(std_dev_inv * (min_edge + d)) -
        erf7(std_dev_inv * d)
    );

    // Invert alpha when `invert` flag is set
    let blur_alpha = select(blur_coverage, 1.0 - blur_coverage, invert != 0u);

    return color * blur_alpha;
}
