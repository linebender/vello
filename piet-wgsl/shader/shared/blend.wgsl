// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Color mixing modes

let MIX_NORMAL = 0u;
let MIX_MULTIPLY = 1u;
let MIX_SCREEN = 2u;
let MIX_OVERLAY = 3u;
let MIX_DARKEN = 4u;
let MIX_LIGHTEN = 5u;
let MIX_COLOR_DODGE = 6u;
let MIX_COLOR_BURN = 7u;
let MIX_HARD_LIGHT = 8u;
let MIX_SOFT_LIGHT = 9u;
let MIX_DIFFERENCE = 10u;
let MIX_EXCLUSION = 11u;
let MIX_HUE = 12u;
let MIX_SATURATION = 13u;
let MIX_COLOR = 14u;
let MIX_LUMINOSITY = 15u;
let MIX_CLIP = 128u;

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
        // MIX_MULTIPLY
        case 1u: {
            b = cb * cs;
        }
        // MIX_SCREEN
        case 2u: {
            b = screen(cb, cs);
        }
        // MIX_OVERLAY
        case 3u: {
            b = hard_light(cs, cb);
        }
        // MIX_DARKEN
        case 4u: {
            b = min(cb, cs);
        }
        // MIX_LIGHTEN
        case 5u: {
            b = max(cb, cs);
        }
        // MIX_COLOR_DODGE
        case 6u: {
            b = vec3(color_dodge(cb.x, cs.x), color_dodge(cb.y, cs.y), color_dodge(cb.z, cs.z));
        }
        // MIX_COLOR_BURN
        case 7u: {
            b = vec3(color_burn(cb.x, cs.x), color_burn(cb.y, cs.y), color_burn(cb.z, cs.z));
        }
        // MIX_HARD_LIGHT
        case 8u: {
            b = hard_light(cb, cs);
        }
        // MIX_SOFT_LIGHT
        case 9u: {
            b = soft_light(cb, cs);
        }
        // MIX_DIFFERENCE
        case 10u: {
            b = abs(cb - cs);
        }
        // MIX_EXCLUSION
        case 11u: {
            b = cb + cs - 2.0 * cb * cs;
        }
        // MIX_HUE
        case 12u: {
            b = set_lum(set_sat(cs, sat(cb)), lum(cb));
        }
        // MIX_SATURATION
        case 13u: {
            b = set_lum(set_sat(cb, sat(cs)), lum(cb));
        }
        // MIX_COLOR
        case 14u: {
            b = set_lum(cs, lum(cb));
        }
        // MIX_LUMINOSITY
        case 15u: {
            b = set_lum(cb, lum(cs));
        }
        default: {
            b = cs;
        }
    }
    return b;
}

// Composition modes

let COMPOSE_CLEAR = 0u;
let COMPOSE_COPY = 1u;
let COMPOSE_DEST = 2u;
let COMPOSE_SRC_OVER = 3u;
let COMPOSE_DEST_OVER = 4u;
let COMPOSE_SRC_IN = 5u;
let COMPOSE_DEST_IN = 6u;
let COMPOSE_SRC_OUT = 7u;
let COMPOSE_DEST_OUT = 8u;
let COMPOSE_SRC_ATOP = 9u;
let COMPOSE_DEST_ATOP = 10u;
let COMPOSE_XOR = 11u;
let COMPOSE_PLUS = 12u;
let COMPOSE_PLUS_LIGHTER = 13u;

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
        // COMPOSE_COPY
        case 1u: {
            fa = 1.0;
            fb = 0.0;
        }
        // COMPOSE_DEST
        case 2u: {
            fa = 0.0;
            fb = 1.0;
        }
        // COMPOSE_SRC_OVER
        case 3u: {
            fa = 1.0;
            fb = 1.0 - as_;
        }
        // COMPOSE_DEST_OVER
        case 4u: {
            fa = 1.0 - ab;
            fb = 1.0;
        }
        // COMPOSE_SRC_IN
        case 5u: {
            fa = ab;
            fb = 0.0;
        }
        // COMPOSE_DEST_IN
        case 6u: {
            fa = 0.0;
            fb = as_;
        }
        // COMPOSE_SRC_OUT
        case 7u: {
            fa = 1.0 - ab;
            fb = 0.0;
        }
        // COMPOSE_DEST_OUT
        case 8u: {
            fa = 0.0;
            fb = 1.0 - as_;
        }
        // COMPOSE_SRC_ATOP
        case 9u: {
            fa = ab;
            fb = 1.0 - as_;
        }
        // COMPOSE_DEST_ATOP
        case 10u: {
            fa = 1.0 - ab;
            fb = as_;
        }
        // COMPOSE_XOR
        case 11u: {
            fa = 1.0 - ab;
            fb = 1.0 - as_;
        }
        // COMPOSE_PLUS
        case 12u: {
            fa = 1.0;
            fb = 1.0;
        }
        // COMPOSE_PLUS_LIGHTER
        case 13u: {
            return min(vec4(1.0), vec4(as_ * cs + ab * cb, as_ + ab));
        }
        default: {}
    }
    let as_fa = as_ * fa;
    let ab_fb = ab * fb;
    let co = as_fa * cs + ab_fb * cb;
    return vec4(co, as_fa + ab_fb);
}

// Apply color mixing and composition. Both input and output colors are
// premultiplied RGB.
fn blend_mix_compose(backdrop: vec4<f32>, src: vec4<f32>, mode: u32) -> vec4<f32> {
    let BLEND_DEFAULT = ((MIX_NORMAL << 8u) | COMPOSE_SRC_OVER);
    let EPSILON = 1e-15;
    if (mode & 0x7fffu) == BLEND_DEFAULT {
        // Both normal+src_over blend and clip case
        return backdrop * (1.0 - src.a) + src;
    }
    // Un-premultiply colors for blending. Max with a small epsilon to avoid NaNs.
    let inv_src_a = 1.0 / max(src.a, EPSILON);
    var cs = src.rgb * inv_src_a;
    let inv_backdrop_a = 1.0 / max(backdrop.a, EPSILON);
    let cb = backdrop.rgb * inv_backdrop_a;
    let mix_mode = mode >> 8u;
    let mixed = blend_mix(cb, cs, mix_mode);
    cs = mix(cs, mixed, backdrop.a);
    let compose_mode = mode & 0xffu;
    if compose_mode == COMPOSE_SRC_OVER {
        let co = mix(backdrop.rgb, cs, src.a);
        return vec4(co, src.a + backdrop.a * (1.0 - src.a));
    } else {
        return blend_compose(cb, cs, backdrop.a, src.a, compose_mode);
    }
}
