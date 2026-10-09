// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sampling and color conversion for YCbCr external textures.

// YCbCr conversion metadata. Must match `vello_common/src/paint.rs`.
const YCBCR_MATRIX_BT601: u32 = 0u;
const YCBCR_MATRIX_BT2020_NCL: u32 = 2u;
const YCBCR_RANGE_LIMITED: u32 = 0u;

fn sample_external_ycbcr_nv12(
    y_texture: texture_2d<f32>,
    uv_texture: texture_2d<f32>,
    quality: u32,
    matrix: u32,
    range: u32,
    coords: vec2<f32>,
    image_offset: vec2<f32>,
    image_size: vec2<f32>,
) -> vec4<f32> {
    let image_max = image_offset + image_size - vec2(1.0);
    let clamped_coords = clamp(coords, image_offset, image_max);
    var y: f32;
    if quality == 0u {
        y = textureLoad(y_texture, vec2<i32>(clamped_coords), 0).r;
    } else {
        let sample_coords = clamped_coords - vec2(0.5);
        let coord_quad = vec4(floor(sample_coords), ceil(sample_coords));
        let coord_frac = fract(sample_coords);
        let a = textureLoad(y_texture, vec2<i32>(clamp(coord_quad.xy, image_offset, image_max)), 0)
                .r;
        let b = textureLoad(y_texture, vec2<i32>(clamp(coord_quad.xw, image_offset, image_max)), 0)
                .r;
        let c = textureLoad(y_texture, vec2<i32>(clamp(coord_quad.zy, image_offset, image_max)), 0)
                .r;
        let d = textureLoad(y_texture, vec2<i32>(clamp(coord_quad.zw, image_offset, image_max)), 0)
                .r;
        y = mix(mix(a, b, coord_frac.y), mix(c, d, coord_frac.y), coord_frac.x);
    }

    let uv_min = floor(image_offset * 0.5);
    let uv_max = floor(image_max * 0.5);
    let uv_coords = vec2<i32>(clamp(floor(clamped_coords * 0.5), uv_min, uv_max));
    let uv = textureLoad(uv_texture, uv_coords, 0).rg;
    let rgb = ycbcr_to_rgb(y, uv.x, uv.y, matrix, range);
    return vec4<f32>(rgb, 1.0);
}

fn ycbcr_to_rgb(y_in: f32, cb_in: f32, cr_in: f32, matrix: u32, range: u32) -> vec3<f32> {
    var y = y_in;
    var cb = cb_in - 0.5;
    var cr = cr_in - 0.5;
    // Limited range, as defined by ITU-T H.273, stores 8-bit Y in 16-235 (219 steps) and
    // Cb/Cr in 16-240 (224 steps); stretch both back to the full range.
    if range == YCBCR_RANGE_LIMITED {
        y = (y - 16.0 / 255.0) * (255.0 / 219.0);
        cb = cb * (255.0 / 224.0);
        cr = cr * (255.0 / 224.0);
    }

    // Luma weights Kr, Kb are defined by ITU-R BT.601, BT.709 and BT.2020 (collected in
    // ITU-T H.273); the standards give the forward transform, inverted here
    // (Kg = 1 - Kr - Kb):
    //   R = Y + 2(1 - Kr) Cr
    //   G = Y - 2Kb(1 - Kb)/Kg Cb - 2Kr(1 - Kr)/Kg Cr
    //   B = Y + 2(1 - Kb) Cb
    // BT.601: Kr = 0.299, Kb = 0.114. BT.2020: Kr = 0.2627, Kb = 0.0593.
    // BT.709 (fallback): Kr = 0.2126, Kb = 0.0722.
    var rgb: vec3<f32>;
    if matrix == YCBCR_MATRIX_BT601 {
        rgb = vec3<f32>(y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb);
    } else if matrix == YCBCR_MATRIX_BT2020_NCL {
        rgb = vec3<f32>(y + 1.4746 * cr, y - 0.16455 * cb - 0.57135 * cr, y + 1.8814 * cb);
    } else {
        rgb = vec3<f32>(y + 1.5748 * cr, y - 0.187324 * cb - 0.468124 * cr, y + 1.8556 * cb);
    }
    return clamp(rgb, vec3(0.0), vec3(1.0));
}
