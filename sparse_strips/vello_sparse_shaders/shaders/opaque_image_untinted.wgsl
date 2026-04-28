// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

const RECT_STRIP_FLAG: u32 = 0x80000000u;
const PAINT_TEXTURE_INDEX_MASK: u32 = 0x03FFFFFFu;

const IMAGE_QUALITY_LOW = 0u;
const IMAGE_QUALITY_MEDIUM = 1u;
const IMAGE_QUALITY_HIGH = 2u;

const TINT_MODE_MULTIPLY: u32 = 1u;

const EXTEND_PAD: u32 = 0u;
const EXTEND_REPEAT: u32 = 1u;
const EXTEND_REFLECT: u32 = 2u;

struct Config {
    width: u32,
    height: u32,
    strip_height: u32,
    alphas_tex_width_bits: u32,
    encoded_paints_tex_width_bits: u32,
    strip_offset_x: i32,
    strip_offset_y: i32,
    ndc_y_negate: u32,
}

struct StripInstance {
    @location(0) xy: u32,
    @location(1) widths_or_rect_height: u32,
    @location(2) col_idx_or_rect_frac: u32,
    @location(3) payload: u32,
    @location(4) paint_and_rect_flag: u32,
    @location(5) depth_index: u32,
}

struct VertexOutput {
    @location(0) sample_xy: vec2<f32>,
    @location(1) @interpolate(flat) paint_tex_idx: u32,
    @builtin(position) position: vec4<f32>,
}

struct EncodedImage {
    quality: u32,
    extend_modes: vec2<u32>,
    image_size: vec2<f32>,
    image_offset: vec2<f32>,
    atlas_index: u32,
    transform: mat2x2<f32>,
    translate: vec2<f32>,
    tint: vec4<f32>,
    tint_mode: u32,
    image_padding: f32,
}

@group(0) @binding(1)
var<uniform> config: Config;

@group(1) @binding(0)
var atlas_texture_array: texture_2d_array<f32>;

@group(2) @binding(0)
var encoded_paints_texture: texture_2d<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    instance: StripInstance,
) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(in_vertex_index & 1u);
    let y = f32(in_vertex_index >> 1u);
    let x0 = instance.xy & 0xffffu;
    let y0 = instance.xy >> 16u;
    let width = instance.widths_or_rect_height & 0xffffu;
    let dense_width = instance.widths_or_rect_height >> 16u;

    var height = config.strip_height;
    if (instance.paint_and_rect_flag & RECT_STRIP_FLAG) != 0u {
        height = dense_width;
    }

    let pix_x = f32(i32(x0) + config.strip_offset_x) + x * f32(width);
    let pix_y = f32(i32(y0) + config.strip_offset_y) + y * f32(height);
    let ndc_x = pix_x * 2.0 / f32(config.width) - 1.0;
    let ndc_y = 1.0 - pix_y * 2.0 / f32(config.height);
    let z = 1.0 - f32(instance.depth_index) / f32(1u << 24u);
    let final_ndc_y = select(ndc_y, -ndc_y, config.ndc_y_negate != 0u);

    let paint_tex_idx = instance.paint_and_rect_flag & PAINT_TEXTURE_INDEX_MASK;
    let encoded_image = unpack_encoded_image(paint_tex_idx);
    let scene_strip_x = instance.payload & 0xffffu;
    let scene_strip_y = instance.payload >> 16u;
    let pos = vec2<f32>(
        f32(scene_strip_x) + x * f32(width),
        f32(scene_strip_y) + y * f32(height)
    );

    out.sample_xy =
        encoded_image.translate + encoded_image.image_offset + encoded_image.transform * pos;
    out.paint_tex_idx = paint_tex_idx;
    out.position = vec4<f32>(ndc_x, final_ndc_y, z, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let encoded_image = unpack_encoded_image(in.paint_tex_idx);
    let image_offset = encoded_image.image_offset;
    let image_size = encoded_image.image_size;
    let local_xy = in.sample_xy - image_offset;
    let offset = 0.00001;
    let extended_xy = vec2<f32>(
        extend_mode(local_xy.x + offset, encoded_image.extend_modes.x, image_size.x),
        extend_mode(local_xy.y + offset, encoded_image.extend_modes.y, image_size.y)
    );

    var sample_color: vec4<f32>;
    if encoded_image.quality == IMAGE_QUALITY_HIGH {
        let final_xy = image_offset + extended_xy;
        sample_color = bicubic_sample(
            atlas_texture_array,
            final_xy,
            i32(encoded_image.atlas_index),
            image_offset,
            image_size,
            encoded_image.extend_modes,
            encoded_image.image_padding,
        );
    } else if encoded_image.quality == IMAGE_QUALITY_MEDIUM {
        let final_xy = image_offset + extended_xy - vec2(0.5);
        sample_color = bilinear_sample(
            atlas_texture_array,
            final_xy,
            i32(encoded_image.atlas_index),
            image_offset,
            image_size,
            encoded_image.extend_modes,
            encoded_image.image_padding,
        );
    } else {
        let final_xy = image_offset + extended_xy;
        sample_color = textureLoad(
            atlas_texture_array,
            vec2<u32>(final_xy),
            i32(encoded_image.atlas_index),
            0,
        );
    }

    return sample_color;
}

fn encoded_paint_coord(flat_idx: u32) -> vec2<u32> {
    return vec2<u32>(
        flat_idx & ((1u << config.encoded_paints_tex_width_bits) - 1u),
        flat_idx >> config.encoded_paints_tex_width_bits
    );
}

fn unpack_encoded_image(paint_tex_idx: u32) -> EncodedImage {
    let texel0 = textureLoad(encoded_paints_texture, encoded_paint_coord(paint_tex_idx), 0);
    let texel1 = textureLoad(encoded_paints_texture, encoded_paint_coord(paint_tex_idx + 1u), 0);
    let texel2 = textureLoad(encoded_paints_texture, encoded_paint_coord(paint_tex_idx + 2u), 0);

    let quality = texel0.x & 0x3u;
    let extend_x = (texel0.x >> 2u) & 0x3u;
    let extend_y = (texel0.x >> 4u) & 0x3u;
    let atlas_index = (texel0.x >> 6u) & 0xFFu;
    let image_size = vec2<f32>(f32(texel0.y >> 16u), f32(texel0.y & 0xFFFFu));
    let image_offset = vec2<f32>(f32(texel0.z >> 16u), f32(texel0.z & 0xFFFFu));
    let transform = mat2x2<f32>(
        vec2<f32>(bitcast<f32>(texel0.w), bitcast<f32>(texel1.x)),
        vec2<f32>(bitcast<f32>(texel1.y), bitcast<f32>(texel1.z))
    );
    let translate = vec2<f32>(bitcast<f32>(texel1.w), bitcast<f32>(texel2.x));
    let packed_tint = texel2.y;
    let tint = select(vec4<f32>(1.0), unpack4x8unorm(packed_tint), packed_tint != 0u);
    let tint_mode = select(TINT_MODE_MULTIPLY, texel2.z, packed_tint != 0u);
    let image_padding = f32(texel2.w);

    return EncodedImage(
        quality,
        vec2<u32>(extend_x, extend_y),
        image_size,
        image_offset,
        atlas_index,
        transform,
        translate,
        tint,
        tint_mode,
        image_padding
    );
}

fn extend_mode(t: f32, mode: u32, max: f32) -> f32 {
    switch mode {
        case EXTEND_PAD: {
            return clamp(t, 0.0, max - 1.0);
        }
        case EXTEND_REPEAT: {
            return extend_mode_normalized(t / max, mode) * max;
        }
        case EXTEND_REFLECT, default: {
            return extend_mode_normalized(t / max, mode) * max;
        }
    }
}

fn extend_mode_normalized(t: f32, mode: u32) -> f32 {
    switch mode {
        case EXTEND_PAD: {
            return clamp(t, 0.0, 1.0);
        }
        case EXTEND_REPEAT: {
            return fract(t);
        }
        case EXTEND_REFLECT, default: {
            return abs(t - 2.0 * round(0.5 * t));
        }
    }
}

fn bilinear_sample(
    tex: texture_2d_array<f32>,
    coords: vec2<f32>,
    atlas_idx: i32,
    image_offset: vec2<f32>,
    image_size: vec2<f32>,
    extend_modes: vec2<u32>,
    image_padding: f32,
) -> vec4<f32> {
    let atlas_max = image_offset + image_size - vec2(1.0);
    let atlas_uv_clamped = clamp(coords, image_offset, atlas_max);
    let uv_quad = vec4(floor(atlas_uv_clamped), ceil(atlas_uv_clamped));
    let uv_frac = fract(coords);
    let a = textureLoad(tex, vec2<i32>(uv_quad.xy), atlas_idx, 0);
    let b = textureLoad(tex, vec2<i32>(uv_quad.xw), atlas_idx, 0);
    let c = textureLoad(tex, vec2<i32>(uv_quad.zy), atlas_idx, 0);
    let d = textureLoad(tex, vec2<i32>(uv_quad.zw), atlas_idx, 0);
    return mix(mix(a, b, uv_frac.y), mix(c, d, uv_frac.y), uv_frac.x);
}

fn bicubic_sample(
    tex: texture_2d_array<f32>,
    coords: vec2<f32>,
    atlas_idx: i32,
    image_offset: vec2<f32>,
    image_size: vec2<f32>,
    extend_modes: vec2<u32>,
    image_padding: f32,
) -> vec4<f32> {
    let atlas_max = image_offset + image_size - vec2(1.0);
    let frac_coords = fract(coords + 0.5);
    let cx = cubic_weights(frac_coords.x);
    let cy = cubic_weights(frac_coords.y);

    let s00 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(-1.5, -1.5), image_offset, atlas_max)), atlas_idx, 0);
    let s10 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(-0.5, -1.5), image_offset, atlas_max)), atlas_idx, 0);
    let s20 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(0.5, -1.5), image_offset, atlas_max)), atlas_idx, 0);
    let s30 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(1.5, -1.5), image_offset, atlas_max)), atlas_idx, 0);

    let s01 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(-1.5, -0.5), image_offset, atlas_max)), atlas_idx, 0);
    let s11 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(-0.5, -0.5), image_offset, atlas_max)), atlas_idx, 0);
    let s21 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(0.5, -0.5), image_offset, atlas_max)), atlas_idx, 0);
    let s31 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(1.5, -0.5), image_offset, atlas_max)), atlas_idx, 0);

    let s02 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(-1.5, 0.5), image_offset, atlas_max)), atlas_idx, 0);
    let s12 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(-0.5, 0.5), image_offset, atlas_max)), atlas_idx, 0);
    let s22 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(0.5, 0.5), image_offset, atlas_max)), atlas_idx, 0);
    let s32 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(1.5, 0.5), image_offset, atlas_max)), atlas_idx, 0);

    let s03 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(-1.5, 1.5), image_offset, atlas_max)), atlas_idx, 0);
    let s13 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(-0.5, 1.5), image_offset, atlas_max)), atlas_idx, 0);
    let s23 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(0.5, 1.5), image_offset, atlas_max)), atlas_idx, 0);
    let s33 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(1.5, 1.5), image_offset, atlas_max)), atlas_idx, 0);

    let row0 = cx.x * s00 + cx.y * s10 + cx.z * s20 + cx.w * s30;
    let row1 = cx.x * s01 + cx.y * s11 + cx.z * s21 + cx.w * s31;
    let row2 = cx.x * s02 + cx.y * s12 + cx.z * s22 + cx.w * s32;
    let row3 = cx.x * s03 + cx.y * s13 + cx.z * s23 + cx.w * s33;
    let result = cy.x * row0 + cy.y * row1 + cy.z * row2 + cy.w * row3;

    let a = clamp(result.a, 0.0, 1.0);
    return vec4<f32>(clamp(result.rgb, vec3(0.0), vec3(a)), a);
}

const MF: array<vec4<f32>, 4> = array<vec4<f32>, 4>(
    vec4<f32>(
        (1.0 / 6.0) / 3.0,
        -(3.0 / 6.0) / 3.0 - 1.0 / 3.0,
        (3.0 / 6.0) / 3.0 + 2.0 * 1.0 / 3.0,
        -(1.0 / 6.0) / 3.0 - 1.0 / 3.0
    ),
    vec4<f32>(
        1.0 - (2.0 / 6.0) / 3.0,
        0.0,
        -3.0 + (12.0 / 6.0) / 3.0 + 1.0 / 3.0,
        2.0 - (9.0 / 6.0) / 3.0 - 1.0 / 3.0
    ),
    vec4<f32>(
        (1.0 / 6.0) / 3.0,
        (3.0 / 6.0) / 3.0 + 1.0 / 3.0,
        3.0 - (15.0 / 6.0) / 3.0 - 2.0 * 1.0 / 3.0,
        -2.0 + (9.0 / 6.0) / 3.0 + 1.0 / 3.0
    ),
    vec4<f32>(
        0.0,
        0.0,
        -1.0 / 3.0,
        (1.0 / 6.0) / 3.0 + 1.0 / 3.0
    )
);

fn cubic_weights(fract: f32) -> vec4<f32> {
    return vec4<f32>(
        single_weight(fract, MF[0][0], MF[0][1], MF[0][2], MF[0][3]),
        single_weight(fract, MF[1][0], MF[1][1], MF[1][2], MF[1][3]),
        single_weight(fract, MF[2][0], MF[2][1], MF[2][2], MF[2][3]),
        single_weight(fract, MF[3][0], MF[3][1], MF[3][2], MF[3][3])
    );
}

fn single_weight(t: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    return t * (t * (t * d + c) + b) + a;
}
