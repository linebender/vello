// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

const RECT_STRIP_FLAG: u32 = 0x80000000u;
const PAINT_TYPE_LINEAR_GRADIENT: u32 = 2u;
const PAINT_TYPE_RADIAL_GRADIENT: u32 = 3u;
const PAINT_TYPE_SWEEP_GRADIENT: u32 = 4u;
const PAINT_TEXTURE_INDEX_MASK: u32 = 0x03FFFFFFu;

const RADIAL_GRADIENT_TYPE_STANDARD: u32 = 0u;
const RADIAL_GRADIENT_TYPE_STRIP: u32 = 1u;
const RADIAL_GRADIENT_TYPE_FOCAL: u32 = 2u;

const PI: f32 = 3.1415926535897932384626433832795028;
const TWO_PI: f32 = 2.0 * PI;
const NEARLY_ZERO_TOLERANCE: f32 = 1.0 / 4096.0;

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
    @location(1) @interpolate(flat) paint_and_rect_flag: u32,
    @builtin(position) position: vec4<f32>,
}

struct LinearGradient {
    extend_mode: u32,
    gradient_start: u32,
    texture_width: u32,
    transform: mat2x2<f32>,
    translate: vec2<f32>,
}

struct RadialGradient {
    extend_mode: u32,
    gradient_start: u32,
    texture_width: u32,
    transform: mat2x2<f32>,
    translate: vec2<f32>,
    bias: f32,
    scale: f32,
    fp0: f32,
    fp1: f32,
    fr1: f32,
    f_focal_x: f32,
    f_is_swapped: u32,
    scaled_r0_squared: f32,
    kind: u32,
}

struct SweepGradient {
    extend_mode: u32,
    gradient_start: u32,
    texture_width: u32,
    transform: mat2x2<f32>,
    translate: vec2<f32>,
    start_angle: f32,
    inv_angle_delta: f32,
}

struct RadialGradientResult {
    t_value: f32,
    is_valid: bool,
}

@group(0) @binding(1)
var<uniform> config: Config;

@group(2) @binding(0)
var encoded_paints_texture: texture_2d<u32>;

@group(3) @binding(0)
var gradient_texture: texture_2d<f32>;

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
    let scene_strip_x = instance.payload & 0xffffu;
    let scene_strip_y = instance.payload >> 16u;

    out.sample_xy = vec2<f32>(
        f32(scene_strip_x) + x * f32(width),
        f32(scene_strip_y) + y * f32(height)
    );
    out.paint_and_rect_flag = instance.paint_and_rect_flag;
    out.position = vec4<f32>(ndc_x, final_ndc_y, z, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let paint_type = (in.paint_and_rect_flag >> 26u) & 0x7u;
    let paint_tex_idx = in.paint_and_rect_flag & PAINT_TEXTURE_INDEX_MASK;

    if paint_type == PAINT_TYPE_LINEAR_GRADIENT {
        let linear_gradient = unpack_linear_gradient(paint_tex_idx);
        let grad_pos = linear_gradient.transform * in.sample_xy + linear_gradient.translate;
        let t_value = grad_pos.x + 0.00001;
        return sample_gradient_lut(
            t_value,
            linear_gradient.extend_mode,
            linear_gradient.gradient_start,
            linear_gradient.texture_width
        );
    } else if paint_type == PAINT_TYPE_RADIAL_GRADIENT {
        let radial_gradient = unpack_radial_gradient(paint_tex_idx);
        let grad_pos = radial_gradient.transform * in.sample_xy + radial_gradient.translate;
        let gradient_result = calculate_radial_gradient(grad_pos, radial_gradient);
        if !gradient_result.is_valid {
            return vec4<f32>(0.0);
        }
        return sample_gradient_lut(
            gradient_result.t_value,
            radial_gradient.extend_mode,
            radial_gradient.gradient_start,
            radial_gradient.texture_width
        );
    } else {
        let sweep_gradient = unpack_sweep_gradient(paint_tex_idx);
        var grad_pos = sweep_gradient.transform * in.sample_xy + sweep_gradient.translate;
        grad_pos = select(grad_pos, vec2(0.0), abs(grad_pos) < vec2(NEARLY_ZERO_TOLERANCE));
        let unit_angle = xy_to_unit_angle(grad_pos.x, grad_pos.y);
        let angle = unit_angle * TWO_PI;
        let t_value = (angle - sweep_gradient.start_angle) * sweep_gradient.inv_angle_delta;
        return sample_gradient_lut(
            t_value,
            sweep_gradient.extend_mode,
            sweep_gradient.gradient_start,
            sweep_gradient.texture_width
        );
    }
}

fn encoded_paint_coord(flat_idx: u32) -> vec2<u32> {
    return vec2<u32>(
        flat_idx & ((1u << config.encoded_paints_tex_width_bits) - 1u),
        flat_idx >> config.encoded_paints_tex_width_bits
    );
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

fn xy_to_unit_angle(x: f32, y: f32) -> f32 {
    let xabs = abs(x);
    let yabs = abs(y);
    let slope = min(xabs, yabs) / max(xabs, yabs);
    let s = slope * slope;
    var phi = slope * (0.15912117063999176025390625 + s * (-5.185396969318389892578125e-2 + s * (2.476101927459239959716796875e-2 + s * (-7.0547382347285747528076171875e-3))));
    phi = select(phi, 0.25 - phi, xabs < yabs);
    phi = select(phi, 0.5 - phi, x < 0.0);
    phi = select(phi, 1.0 - phi, y < 0.0);
    phi = select(phi, 0.0, phi != phi);
    return phi;
}

fn sample_gradient_lut(
    t_value: f32,
    extend_mode: u32,
    gradient_start: u32,
    texture_width: u32,
) -> vec4<f32> {
    let clamped_t = extend_mode_normalized(t_value, extend_mode);
    let t_offset = u32(clamped_t * f32(texture_width - 1u));
    let flat_coord = gradient_start + t_offset;
    let gradient_tex_width = textureDimensions(gradient_texture).x;
    let tex_x = flat_coord % gradient_tex_width;
    let tex_y = flat_coord / gradient_tex_width;
    return textureLoad(gradient_texture, vec2<u32>(tex_x, tex_y), 0);
}

fn unpack_linear_gradient(paint_tex_idx: u32) -> LinearGradient {
    let texel0 = textureLoad(encoded_paints_texture, encoded_paint_coord(paint_tex_idx), 0);
    let texel1 = textureLoad(encoded_paints_texture, encoded_paint_coord(paint_tex_idx + 1u), 0);
    let texture_width_and_extend_mode = unpack_texture_width_and_extend_mode(texel0.x);
    let texture_width = texture_width_and_extend_mode.x;
    let extend_mode = texture_width_and_extend_mode.y;
    let gradient_start = texel0.y;
    let transform = mat2x2<f32>(
        vec2<f32>(bitcast<f32>(texel0.z), bitcast<f32>(texel0.w)),
        vec2<f32>(bitcast<f32>(texel1.x), bitcast<f32>(texel1.y))
    );
    let translate = vec2<f32>(bitcast<f32>(texel1.z), bitcast<f32>(texel1.w));
    return LinearGradient(extend_mode, gradient_start, texture_width, transform, translate);
}

fn calculate_radial_gradient(
    grad_pos: vec2<f32>,
    radial_gradient: RadialGradient,
) -> RadialGradientResult {
    let x_pos = grad_pos.x;
    let y_pos = grad_pos.y;

    var t_value: f32;
    var is_valid: bool;

    switch radial_gradient.kind {
        case RADIAL_GRADIENT_TYPE_STANDARD: {
            let radius = sqrt(x_pos * x_pos + y_pos * y_pos);
            t_value = radial_gradient.bias + radial_gradient.scale * radius;
            is_valid = true;
        }
        case RADIAL_GRADIENT_TYPE_STRIP: {
            let p1 = radial_gradient.scaled_r0_squared - y_pos * y_pos;
            is_valid = p1 >= 0.0;
            if is_valid {
                t_value = x_pos + sqrt(p1);
            } else {
                t_value = 0.0;
            }
        }
        case RADIAL_GRADIENT_TYPE_FOCAL, default: {
            var t = 0.0;
            let fp0 = radial_gradient.fp0;
            let fp1 = radial_gradient.fp1;
            let fr1 = radial_gradient.fr1;
            let f_focal_x = radial_gradient.f_focal_x;
            let is_swapped = radial_gradient.f_is_swapped;
            let is_focal_on_circle = abs(1.0 - fr1) <= NEARLY_ZERO_TOLERANCE;
            let is_well_behaved = !is_focal_on_circle && fr1 > 1.0;
            let is_natively_focal = abs(f_focal_x) <= NEARLY_ZERO_TOLERANCE;

            is_valid = true;

            if is_focal_on_circle {
                t = x_pos + y_pos * y_pos / x_pos;
                is_valid = t >= 0.0 && x_pos != 0.0;
            } else if is_well_behaved {
                t = sqrt(x_pos * x_pos + y_pos * y_pos) - x_pos * fp0;
            } else {
                let xx = x_pos * x_pos;
                let yy = y_pos * y_pos;
                let discriminant = xx - yy;

                if is_swapped != 0u || (1.0 - f_focal_x < 0.0) {
                    t = -sqrt(discriminant) - x_pos * fp0;
                } else {
                    t = sqrt(discriminant) - x_pos * fp0;
                }

                is_valid = discriminant >= 0.0 && t >= 0.0;
            }

            if is_valid {
                if 1.0 - f_focal_x < 0.0 {
                    t = -t;
                }

                if !is_natively_focal {
                    t = t + fp1;
                }

                if is_swapped != 0u {
                    t = 1.0 - t;
                }
            }

            t_value = t;
        }
    }

    return RadialGradientResult(t_value, is_valid);
}

fn unpack_radial_gradient(paint_tex_idx: u32) -> RadialGradient {
    let texel0 = textureLoad(encoded_paints_texture, encoded_paint_coord(paint_tex_idx), 0);
    let texel1 = textureLoad(encoded_paints_texture, encoded_paint_coord(paint_tex_idx + 1u), 0);
    let texel2 = textureLoad(encoded_paints_texture, encoded_paint_coord(paint_tex_idx + 2u), 0);
    let texel3 = textureLoad(encoded_paints_texture, encoded_paint_coord(paint_tex_idx + 3u), 0);
    let texture_width_and_extend_mode = unpack_texture_width_and_extend_mode(texel0.x);
    let texture_width = texture_width_and_extend_mode.x;
    let extend_mode = texture_width_and_extend_mode.y;
    let gradient_start = texel0.y;
    let transform = mat2x2<f32>(
        vec2<f32>(bitcast<f32>(texel0.z), bitcast<f32>(texel0.w)),
        vec2<f32>(bitcast<f32>(texel1.x), bitcast<f32>(texel1.y))
    );
    let translate = vec2<f32>(bitcast<f32>(texel1.z), bitcast<f32>(texel1.w));
    let kind_and_swapped = unpack_radial_kind_and_swapped(texel2.x);
    let kind = kind_and_swapped.x;
    let f_is_swapped = kind_and_swapped.y;
    let bias = bitcast<f32>(texel2.y);
    let scale = bitcast<f32>(texel2.z);
    let fp0 = bitcast<f32>(texel2.w);
    let fp1 = bitcast<f32>(texel3.x);
    let fr1 = bitcast<f32>(texel3.y);
    let f_focal_x = bitcast<f32>(texel3.z);
    let scaled_r0_squared = bitcast<f32>(texel3.w);
    return RadialGradient(
        extend_mode,
        gradient_start,
        texture_width,
        transform,
        translate,
        bias,
        scale,
        fp0,
        fp1,
        fr1,
        f_focal_x,
        f_is_swapped,
        scaled_r0_squared,
        kind
    );
}

fn unpack_sweep_gradient(paint_tex_idx: u32) -> SweepGradient {
    let texel0 = textureLoad(encoded_paints_texture, encoded_paint_coord(paint_tex_idx), 0);
    let texel1 = textureLoad(encoded_paints_texture, encoded_paint_coord(paint_tex_idx + 1u), 0);
    let texel2 = textureLoad(encoded_paints_texture, encoded_paint_coord(paint_tex_idx + 2u), 0);
    let texture_width_and_extend_mode = unpack_texture_width_and_extend_mode(texel0.x);
    let texture_width = texture_width_and_extend_mode.x;
    let extend_mode = texture_width_and_extend_mode.y;
    let gradient_start = texel0.y;
    let transform = mat2x2<f32>(
        vec2<f32>(bitcast<f32>(texel0.z), bitcast<f32>(texel0.w)),
        vec2<f32>(bitcast<f32>(texel1.x), bitcast<f32>(texel1.y))
    );
    let translate = vec2<f32>(bitcast<f32>(texel1.z), bitcast<f32>(texel1.w));
    let start_angle = bitcast<f32>(texel2.x);
    let inv_angle_delta = bitcast<f32>(texel2.y);
    return SweepGradient(
        extend_mode,
        gradient_start,
        texture_width,
        transform,
        translate,
        start_angle,
        inv_angle_delta
    );
}

fn unpack_texture_width_and_extend_mode(packed: u32) -> vec2<u32> {
    let texture_width = packed & 0x0FFFFFFFu;
    let extend_mode = (packed >> 30u) & 3u;
    return vec2<u32>(texture_width, extend_mode);
}

fn unpack_radial_kind_and_swapped(packed: u32) -> vec2<u32> {
    let kind = packed & 0x3u;
    let f_is_swapped = (packed >> 2u) & 0x1u;
    return vec2<u32>(kind, f_is_swapped);
}
