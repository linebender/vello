// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

struct Config {
    // Width of a wide tile (matching `WideTile::WIDTH`).
    wide_tile_width: u32,
    // Height of a wide tile (matching `WideTile::HEIGHT`).
    wide_tile_height: u32,
    // Height of the slot texture.
    slot_texture_height: u32,
    // Height of the final target texture.
    final_target_height: u32,
    // Height of the blend texture.
    blend_texture_height: u32,
}

struct BlendCommand {
    // [x, y] packed as u16's
    // x, y — coordinates of the top left of the source wide tile
    @location(0) xy_src: u32,
    // [x, y] packed as u16's
    // x, y — coordinates of the top left of the destination wide tile
    @location(1) xy_dst: u32,
    // Bits 0-7: opacity
    // Bits 8-11: compose
    // Bits 12-15: mix
    // Bits 16: source texture (TODO: Consider passing slot_ix alone)
    //       0 = slots of ix=0
    //       1 = slots of ix=1
    // Bits 17-18: dest texture
    //       0 = slots of ix=0
    //       1 = slots of ix=1
    //       2 = final target
    // Bits 19-26: blend slot index
    @location(2) payload: u32,
}

struct VertexOutput {
    // Normalized device coordinates (NDC) for the current vertex
    @builtin(position) position: vec4<f32>,
    // Texture coordinates for the current fragment
    @location(0) src_tex_coord: vec2<f32>,
    @location(1) dst_tex_coord: vec2<f32>,
    // See `BlendCommand` documentation.
    @location(2) payload: u32,
}

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var slot_texture_0: texture_2d<f32>;

@group(0) @binding(2)
var slot_texture_1: texture_2d<f32>;

@group(0) @binding(3)
var final_target: texture_2d<f32>;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    command: BlendCommand,
) -> VertexOutput {
    var out: VertexOutput;
    out.payload = command.payload;

    // Map vertex_index (0-3) to quad corners:
    // 0 → (0,0), 1 → (1,0), 2 → (0,1), 3 → (1,1)
    let x = f32(in_vertex_index & 1u);
    let y = f32(in_vertex_index >> 1u);

    // Calculate `position` for output.
    {
        // Extract bits 19-26 for blend slot index
        let blend_slot_ix = (command.payload >> 19u) & 0xffu;

        // Calculate the y-position based on the slot index
        let slot_y_offset = f32(blend_slot_ix * config.wide_tile_height);

        // Scale to match slot dimensions
        let pix_x = x * f32(config.wide_tile_width);
        let pix_y = slot_y_offset + y * f32(config.wide_tile_height);

        // Convert to NDC
        let ndc_x = pix_x * 2.0 / f32(config.wide_tile_width) - 1.0;
        let ndc_y = 1.0 - pix_y * 2.0 / f32(config.blend_texture_height);

        out.position = vec4(ndc_x, ndc_y, 0.0, 1.0);
    }

    // Calculate `src_tex_coord` for the source texture.
    {
        let src_x0 = f32(command.xy_src & 0xffffu);
        let src_y0 = f32(command.xy_src >> 16u);

        let src_x = src_x0 + x * f32(config.wide_tile_width);
        let src_y = src_y0 + y * f32(config.wide_tile_height);

        out.src_tex_coord = vec2f(src_x, src_y);
    }

    // Calculate `dst_tex_coord` for the destination texture.
    {
        let dst_x0 = f32(command.xy_dst & 0xffffu);
        let dst_y0 = f32(command.xy_dst >> 16u);

        let dst_texture_ix = (command.payload >> 17u) & 3u;

        let dst_height = f32(dst_texture_ix != 2u) * f32(config.wide_tile_height) + f32(dst_texture_ix == 2) * f32(config.final_target_height);

        let dst_x = dst_x0 + x * f32(config.wide_tile_width);
        let dst_y = dst_y0 + y * f32(dst_height);

        out.dst_tex_coord = vec2f(dst_x, dst_y);
    }

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var bg_color: vec4<f32>;
    var fg_color: vec4<f32>;

    // Calculate `fg_color` of the foreground texture.
    {
        let src_texture_ix = (in.payload >> 16u) & 1u;
        let src = vec2u(u32(floor(in.src_tex_coord.x)), u32(floor(in.src_tex_coord.y)));

        if src_texture_ix == 0u {
            fg_color = textureLoad(slot_texture_0, src, 0);
        } else {
            fg_color = textureLoad(slot_texture_1, src, 0);
        }
    }

    // Calculate `bg_color` of the background texture.
    {
        let dst_texture_ix = (in.payload >> 17u) & 3u;
        let dst = vec2u(u32(floor(in.dst_tex_coord.x)), u32(floor(in.dst_tex_coord.y)));

        if dst_texture_ix == 0u {
            bg_color = textureLoad(slot_texture_0, dst, 0);
        } else if dst_texture_ix == 1u {
            bg_color = textureLoad(slot_texture_1, dst, 0);
        } else {
            bg_color = textureLoad(final_target, dst, 0);
        }
    }

    let opacity = f32(in.payload & 0xFFu) / 255.0;
    let mixed = blend_mix_compose(bg_color, fg_color, in.payload >> 8u);
    return mixed * opacity;
}

// Color mixing modes

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
const MIX_CLIP = 16u;

fn screen(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
    return cb + cs - (cb * cs);
}

fn color_dodge(cb: f32, cs: f32) -> f32 {
    if cb == 0.0 {
        return 0.0;
    } else if cs == 1.0 {
        return 1.0;
    } else {
        return min(1.0, cb / (1.0 - cs));
    }
}

fn color_burn(cb: f32, cs: f32) -> f32 {
    if cb == 1.0 {
        return 1.0;
    } else if cs == 0.0 {
        return 0.0;
    } else {
        return 1.0 - min(1.0, (1.0 - cb) / cs);
    }
}

fn hard_light(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
    return select(
        screen(cb, 2.0 * cs - 1.0),
        cb * 2.0 * cs,
        cs <= vec3(0.5)
    );
}

fn soft_light(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
    let d = select(
        sqrt(cb),
        ((16.0 * cb - 12.0) * cb + 4.0) * cb,
        cb <= vec3(0.25)
    );
    return select(
        cb + (2.0 * cs - 1.0) * (d - cb),
        cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb),
        cs <= vec3(0.5)
    );
}

fn sat(c: vec3<f32>) -> f32 {
    return max(c.x, max(c.y, c.z)) - min(c.x, min(c.y, c.z));
}

fn lum(c: vec3<f32>) -> f32 {
    let f = vec3(0.3, 0.59, 0.11);
    return dot(c, f);
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
        } else {
            if r <= b {
                set_sat_inner(&r, &b, &g, s);
            } else {
                set_sat_inner(&b, &r, &g, s);
            }
        }
    } else {
        if r <= b {
            set_sat_inner(&g, &r, &b, s);
        } else {
            if g <= b {
                set_sat_inner(&g, &b, &r, s);
            } else {
                set_sat_inner(&b, &g, &r, s);
            }
        }
    }
    return vec3(r, g, b);
}

// Blends two RGB colors together. The colors are assumed to be in sRGB
// color space, and this function does not take alpha into account.
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

// Composition modes

const COMPOSE_CLEAR = 0u;
const COMPOSE_COPY = 1u;
const COMPOSE_DEST = 2u;
const COMPOSE_SRC_OVER = 3u;
const COMPOSE_DEST_OVER = 4u;
const COMPOSE_SRC_IN = 5u;
const COMPOSE_DEST_IN = 6u;
const COMPOSE_SRC_OUT = 7u;
const COMPOSE_DEST_OUT = 8u;
const COMPOSE_SRC_ATOP = 9u;
const COMPOSE_DEST_ATOP = 10u;
const COMPOSE_XOR = 11u;
const COMPOSE_PLUS = 12u;
const COMPOSE_PLUS_LIGHTER = 13u;

// Apply general compositing operation.
// Inputs are separated colors and alpha, output is premultiplied.
fn blend_compose(
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
    // Modes like COMPOSE_PLUS can generate alpha > 1.0, so clamp.
    return vec4(co, min(as_fa + ab_fb, 1.0));
}

// Apply color mixing and composition. Both input and output colors are
// premultiplied RGB.
fn blend_mix_compose(backdrop: vec4<f32>, src: vec4<f32>, mode: u32) -> vec4<f32> {
    let BLEND_DEFAULT = ((MIX_NORMAL << 4u) | COMPOSE_SRC_OVER);
    let EPSILON = 1e-15;
    if (mode & 0xffu) == BLEND_DEFAULT {
        // Both normal+src_over blend and clip case
        return backdrop * (1.0 - src.a) + src;
    }
    // Un-premultiply colors for blending. Max with a small epsilon to avoid NaNs.
    let inv_src_a = 1.0 / max(src.a, EPSILON);
    var cs = src.rgb * inv_src_a;
    let inv_backdrop_a = 1.0 / max(backdrop.a, EPSILON);
    let cb = backdrop.rgb * inv_backdrop_a;
    let mix_mode = (mode >> 4u) & 0xfu;
    let mixed = blend_mix(cb, cs, mix_mode);
    cs = mix(cs, mixed, backdrop.a);
    let compose_mode = mode & 0xfu;
    if compose_mode == COMPOSE_SRC_OVER {
        let co = mix(backdrop.rgb, cs, src.a);
        return vec4(co, src.a + backdrop.a * (1.0 - src.a));
    } else {
        return blend_compose(cb, cs, backdrop.a, src.a, compose_mode);
    }
}

