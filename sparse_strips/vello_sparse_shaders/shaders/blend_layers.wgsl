// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Blend two rendered layer atlas regions through scratch.

const COMPOSE_CLEAR: u32 = 0u;
const COMPOSE_COPY: u32 = 1u;
const COMPOSE_DEST: u32 = 2u;
const COMPOSE_SRC_OVER: u32 = 3u;
const COMPOSE_DEST_OVER: u32 = 4u;
const COMPOSE_SRC_IN: u32 = 5u;
const COMPOSE_DEST_IN: u32 = 6u;
const COMPOSE_SRC_OUT: u32 = 7u;
const COMPOSE_DEST_OUT: u32 = 8u;
const COMPOSE_SRC_ATOP: u32 = 9u;
const COMPOSE_DEST_ATOP: u32 = 10u;
const COMPOSE_XOR: u32 = 11u;
const COMPOSE_PLUS: u32 = 12u;
const COMPOSE_PLUS_LIGHTER: u32 = 13u;

const MIX_NORMAL = 0u;
const MIX_MULTIPLY = 1u;
const MIX_SCREEN = 2u;
const MIX_OVERLAY = 3u;
const MIX_DARKEN = 4u;
const MIX_LIGHTEN = 5u;
const MIX_COLOR_DODGE = 6u;
const MIX_COLOR_BURN = 7u;
const MIX_HARD_LIGHT = 8u;
const MIX_SOFT_LIGHT = 9u;
const MIX_DIFFERENCE = 10u;
const MIX_EXCLUSION = 11u;
const MIX_HUE = 12u;
const MIX_SATURATION = 13u;
const MIX_COLOR = 14u;
const MIX_LUMINOSITY = 15u;

struct BlendInstance {
    @location(0) dest_origin: vec2<u32>,
    @location(1) source_origin: vec2<u32>,
    @location(2) size: vec2<u32>,
    @location(3) texture_indices: vec2<u32>,
    @location(4) blend_mode: vec2<u32>,
    @location(5) opacity: u32,
    @location(6) target_size: vec2<u32>,
    @location(7) bbox_origin: vec2<u32>,
    @location(8) source_scene_origin: vec2<u32>,
    @location(9) source_size: vec2<u32>,
}

struct VertexOutput {
    @location(0) dest_xy: vec2<f32>,
    @location(1) source_xy: vec2<f32>,
    @location(2) source_local: vec2<f32>,
    @location(3) @interpolate(flat) source_size: vec2<u32>,
    @location(4) @interpolate(flat) texture_indices: vec2<u32>,
    @location(5) @interpolate(flat) blend_mode: vec2<u32>,
    @location(6) @interpolate(flat) opacity: u32,
    @builtin(position) position: vec4<f32>,
}

@group(0) @binding(0)
var layer_texture_0: texture_2d<f32>;

@group(0) @binding(1)
var layer_texture_1: texture_2d<f32>;

@group(1) @binding(0)
var scratch_texture: texture_2d<f32>;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    instance: BlendInstance,
) -> VertexOutput {
    let x = f32(vertex_index & 1u);
    let y = f32(vertex_index >> 1u);
    let local = vec2<f32>(x, y) * vec2<f32>(instance.size);
    let dest_xy = vec2<f32>(instance.dest_origin) + local;
    let scene_xy = vec2<f32>(instance.bbox_origin) + local;
    let source_local = scene_xy - vec2<f32>(instance.source_scene_origin);

    var out: VertexOutput;
    out.dest_xy = dest_xy;
    out.source_xy = vec2<f32>(instance.source_origin) + source_local;
    out.source_local = source_local;
    out.source_size = instance.source_size;
    out.texture_indices = instance.texture_indices;
    out.blend_mode = instance.blend_mode;
    out.opacity = instance.opacity;

    let ndc_x = dest_xy.x * 2.0 / f32(instance.target_size.x) - 1.0;
    let ndc_y = 1.0 - dest_xy.y * 2.0 / f32(instance.target_size.y);
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    return out;
}

fn load_layer(texture_index: u32, xy: vec2<i32>) -> vec4<f32> {
    if texture_index == 0u {
        return textureLoad(layer_texture_0, xy, 0);
    }
    return textureLoad(layer_texture_1, xy, 0);
}

@fragment
fn fs_main(
    @location(0) dest_xy: vec2<f32>,
    @location(1) source_xy: vec2<f32>,
    @location(2) source_local: vec2<f32>,
    @location(3) @interpolate(flat) source_size: vec2<u32>,
    @location(4) @interpolate(flat) texture_indices: vec2<u32>,
    @location(5) @interpolate(flat) blend_mode: vec2<u32>,
    @location(6) @interpolate(flat) opacity: u32,
) -> @location(0) vec4<f32> {
    let backdrop = load_layer(texture_indices.x, vec2<i32>(dest_xy));
    let source_opacity = f32(opacity) * (1.0 / 255.0);
    var source = vec4<f32>(0.0);
    if source_local.x >= 0.0
        && source_local.y >= 0.0
        && source_local.x < f32(source_size.x)
        && source_local.y < f32(source_size.y)
    {
        source = source_opacity * load_layer(texture_indices.y, vec2<i32>(source_xy));
    }
    return blend_mix_compose(backdrop, source, blend_mode.y, blend_mode.x);
}

@fragment
fn fs_copy(
    @location(0) dest_xy: vec2<f32>,
) -> @location(0) vec4<f32> {
    return textureLoad(scratch_texture, vec2<i32>(dest_xy), 0);
}

fn blend_mix_compose(backdrop: vec4<f32>, src: vec4<f32>, compose_mode: u32, mix_mode: u32) -> vec4<f32> {
    let BLEND_DEFAULT = ((MIX_NORMAL << 8u) | COMPOSE_SRC_OVER);
    let mode = ((mix_mode << 8u) | compose_mode);
    if mode == BLEND_DEFAULT {
        return backdrop * (1.0 - src.a) + src;
    }

    let EPSILON = 1e-15;
    let inv_src_a = 1.0 / max(src.a, EPSILON);
    var cs = src.rgb * inv_src_a;
    let inv_backdrop_a = 1.0 / max(backdrop.a, EPSILON);
    let cb = backdrop.rgb * inv_backdrop_a;
    let mixed = blend_mix(cb, cs, mix_mode);
    cs = mix(cs, mixed, backdrop.a);

    if compose_mode == COMPOSE_SRC_OVER {
        let co = mix(backdrop.rgb, cs, src.a);
        return vec4(co, src.a + backdrop.a * (1.0 - src.a));
    }
    return blend_compose_unpremul(cb, cs, backdrop.a, src.a, compose_mode);
}

fn blend_compose_unpremul(
    cb: vec3<f32>,
    cs: vec3<f32>,
    ab: f32,
    as_: f32,
    mode: u32
) -> vec4<f32> {
    var fa = 0.0;
    var fb = 0.0;
    switch mode {
        case COMPOSE_COPY: {
            fa = 1.0;
            fb = 0.0;
        }
        case COMPOSE_DEST: {
            fa = 0.0;
            fb = 1.0;
        }
        case COMPOSE_SRC_OVER: {
            fa = 1.0;
            fb = 1.0 - as_;
        }
        case COMPOSE_DEST_OVER: {
            fa = 1.0 - ab;
            fb = 1.0;
        }
        case COMPOSE_SRC_IN: {
            fa = ab;
            fb = 0.0;
        }
        case COMPOSE_DEST_IN: {
            fa = 0.0;
            fb = as_;
        }
        case COMPOSE_SRC_OUT: {
            fa = 1.0 - ab;
            fb = 0.0;
        }
        case COMPOSE_DEST_OUT: {
            fa = 0.0;
            fb = 1.0 - as_;
        }
        case COMPOSE_SRC_ATOP: {
            fa = ab;
            fb = 1.0 - as_;
        }
        case COMPOSE_DEST_ATOP: {
            fa = 1.0 - ab;
            fb = as_;
        }
        case COMPOSE_XOR: {
            fa = 1.0 - ab;
            fb = 1.0 - as_;
        }
        case COMPOSE_PLUS: {
            fa = 1.0;
            fb = 1.0;
        }
        case COMPOSE_PLUS_LIGHTER: {
            return min(vec4(1.0), vec4(as_ * cs + ab * cb, as_ + ab));
        }
        default: {}
    }
    let as_fa = as_ * fa;
    let ab_fb = ab * fb;
    let co = as_fa * cs + ab_fb * cb;
    return vec4(co, min(as_fa + ab_fb, 1.0));
}

fn screen(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
    return cb + cs - (cb * cs);
}

fn color_dodge(cb: f32, cs: f32) -> f32 {
    if cb == 0.0 {
        return 0.0;
    } else if cs == 1.0 {
        return 1.0;
    }
    return min(1.0, cb / (1.0 - cs));
}

fn color_burn(cb: f32, cs: f32) -> f32 {
    if cb == 1.0 {
        return 1.0;
    } else if cs == 0.0 {
        return 0.0;
    }
    return 1.0 - min(1.0, (1.0 - cb) / cs);
}

fn hard_light(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
    return select(screen(cb, 2.0 * cs - 1.0), cb * 2.0 * cs, cs <= vec3(0.5));
}

fn soft_light(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
    let d = select(sqrt(cb), ((16.0 * cb - 12.0) * cb + 4.0) * cb, cb <= vec3(0.25));
    return select(cb + (2.0 * cs - 1.0) * (d - cb), cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb), cs <= vec3(0.5));
}

fn sat(c: vec3<f32>) -> f32 {
    return max(c.x, max(c.y, c.z)) - min(c.x, min(c.y, c.z));
}

fn lum(c: vec3<f32>) -> f32 {
    return dot(c, vec3(0.3, 0.59, 0.11));
}

fn clip_color(c_in: vec3<f32>) -> vec3<f32> {
    var c = c_in;
    let l = lum(c);
    let n = min(c.x, min(c.y, c.z));
    let x = max(c.x, max(c.y, c.z));
    if n < 0.0 {
        c = l + (((c - l) * l) / (l - n));
    }
    if x > 1.0 {
        c = l + (((c - l) * (1.0 - l)) / (x - l));
    }
    return c;
}

fn set_lum(c: vec3<f32>, l: f32) -> vec3<f32> {
    return clip_color(c + (l - lum(c)));
}

fn set_sat_inner(
    cmin: ptr<function, f32>,
    cmid: ptr<function, f32>,
    cmax: ptr<function, f32>,
    s: f32
) {
    if *cmax > *cmin {
        *cmid = ((*cmid - *cmin) * s) / (*cmax - *cmin);
        *cmax = s;
    } else {
        *cmid = 0.0;
        *cmax = 0.0;
    }
    *cmin = 0.0;
}

fn set_sat(c: vec3<f32>, s: f32) -> vec3<f32> {
    var r = c.r;
    var g = c.g;
    var b = c.b;
    if r <= g {
        if g <= b {
            set_sat_inner(&r, &g, &b, s);
        } else if r <= b {
            set_sat_inner(&r, &b, &g, s);
        } else {
            set_sat_inner(&b, &r, &g, s);
        }
    } else if r <= b {
        set_sat_inner(&g, &r, &b, s);
    } else if g <= b {
        set_sat_inner(&g, &b, &r, s);
    } else {
        set_sat_inner(&b, &g, &r, s);
    }
    return vec3(r, g, b);
}

fn blend_mix(cb: vec3<f32>, cs: vec3<f32>, mode: u32) -> vec3<f32> {
    var b = vec3(0.0);
    switch mode {
        case MIX_MULTIPLY: {
            b = cb * cs;
        }
        case MIX_SCREEN: {
            b = screen(cb, cs);
        }
        case MIX_OVERLAY: {
            b = hard_light(cs, cb);
        }
        case MIX_DARKEN: {
            b = min(cb, cs);
        }
        case MIX_LIGHTEN: {
            b = max(cb, cs);
        }
        case MIX_COLOR_DODGE: {
            b = vec3(color_dodge(cb.x, cs.x), color_dodge(cb.y, cs.y), color_dodge(cb.z, cs.z));
        }
        case MIX_COLOR_BURN: {
            b = vec3(color_burn(cb.x, cs.x), color_burn(cb.y, cs.y), color_burn(cb.z, cs.z));
        }
        case MIX_HARD_LIGHT: {
            b = hard_light(cb, cs);
        }
        case MIX_SOFT_LIGHT: {
            b = soft_light(cb, cs);
        }
        case MIX_DIFFERENCE: {
            b = abs(cb - cs);
        }
        case MIX_EXCLUSION: {
            b = cb + cs - 2.0 * cb * cs;
        }
        case MIX_HUE: {
            b = set_lum(set_sat(cs, sat(cb)), lum(cb));
        }
        case MIX_SATURATION: {
            b = set_lum(set_sat(cb, sat(cs)), lum(cb));
        }
        case MIX_COLOR: {
            b = set_lum(cs, lum(cb));
        }
        case MIX_LUMINOSITY: {
            b = set_lum(cb, lum(cs));
        }
        default: {
            b = cs;
        }
    }
    return b;
}
