// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Fine rasterizer. This can run in simple (just path rendering) and full
// modes, controllable by #define.

// This is a cut'n'paste w/ backdrop.
struct Tile {
    backdrop: i32,
    segments: u32,
}

#import segment
#import config

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> tiles: array<Tile>;

@group(0) @binding(2)
var<storage> segments: array<Segment>;

#ifdef full

#import blend
#import ptcl

let GRADIENT_WIDTH = 512;

@group(0) @binding(3)
var output: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(4)
var<storage> ptcl: array<u32>;

@group(0) @binding(5)
var gradients: texture_2d<f32>;

@group(0) @binding(6)
var<storage> info: array<u32>;

@group(0) @binding(7)
var image_atlas: texture_2d<f32>;

fn read_fill(cmd_ix: u32) -> CmdFill {
    let tile = ptcl[cmd_ix + 1u];
    let backdrop = i32(ptcl[cmd_ix + 2u]);
    return CmdFill(tile, backdrop);
}

fn read_stroke(cmd_ix: u32) -> CmdStroke {
    let tile = ptcl[cmd_ix + 1u];
    let half_width = bitcast<f32>(ptcl[cmd_ix + 2u]);
    return CmdStroke(tile, half_width);
}

fn read_color(cmd_ix: u32) -> CmdColor {
    let rgba_color = ptcl[cmd_ix + 1u];
    return CmdColor(rgba_color);
}

fn read_lin_grad(cmd_ix: u32) -> CmdLinGrad {
    let index = ptcl[cmd_ix + 1u];
    let info_offset = ptcl[cmd_ix + 2u];
    let line_x = bitcast<f32>(info[info_offset]);
    let line_y = bitcast<f32>(info[info_offset + 1u]);
    let line_c = bitcast<f32>(info[info_offset + 2u]);
    return CmdLinGrad(index, line_x, line_y, line_c);
}

fn read_rad_grad(cmd_ix: u32) -> CmdRadGrad {
    let index = ptcl[cmd_ix + 1u];
    let info_offset = ptcl[cmd_ix + 2u];
    let m0 = bitcast<f32>(info[info_offset]);
    let m1 = bitcast<f32>(info[info_offset + 1u]);
    let m2 = bitcast<f32>(info[info_offset + 2u]);
    let m3 = bitcast<f32>(info[info_offset + 3u]);
    let matrx = vec4(m0, m1, m2, m3);
    let xlat = vec2(bitcast<f32>(info[info_offset + 4u]), bitcast<f32>(info[info_offset + 5u]));
    let c1 = vec2(bitcast<f32>(info[info_offset + 6u]), bitcast<f32>(info[info_offset + 7u]));
    let ra = bitcast<f32>(info[info_offset + 8u]);
    let roff = bitcast<f32>(info[info_offset + 9u]);
    return CmdRadGrad(index, matrx, xlat, c1, ra, roff);
}

fn read_image(cmd_ix: u32) -> CmdImage {
    let info_offset = ptcl[cmd_ix + 1u];
    let m0 = bitcast<f32>(info[info_offset]);
    let m1 = bitcast<f32>(info[info_offset + 1u]);
    let m2 = bitcast<f32>(info[info_offset + 2u]);
    let m3 = bitcast<f32>(info[info_offset + 3u]);
    let matrx = vec4(m0, m1, m2, m3);
    let xlat = vec2(bitcast<f32>(info[info_offset + 4u]), bitcast<f32>(info[info_offset + 5u]));
    let xy = info[info_offset + 6u];
    let width_height = info[info_offset + 7u];
    // The following are not intended to be bitcasts
    let x = f32(xy >> 16u);
    let y = f32(xy & 0xffffu);
    let width = f32(width_height >> 16u);
    let height = f32(width_height & 0xffffu);
    return CmdImage(matrx, xlat, vec2(x, y), vec2(width, height));
}

fn read_end_clip(cmd_ix: u32) -> CmdEndClip {
    let blend = ptcl[cmd_ix + 1u];
    let alpha = bitcast<f32>(ptcl[cmd_ix + 2u]);
    return CmdEndClip(blend, alpha);
}

#else

@group(0) @binding(3)
var output: texture_storage_2d<r8, write>;

#endif

let PIXELS_PER_THREAD = 4u;

fn fill_path(tile: Tile, xy: vec2<f32>, even_odd: bool) -> array<f32, PIXELS_PER_THREAD> {
    var area: array<f32, PIXELS_PER_THREAD>;
    let backdrop_f = f32(tile.backdrop);
    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
        area[i] = backdrop_f;
    }
    var segment_ix = tile.segments;
    while segment_ix != 0u {
        let segment = segments[segment_ix];
        let y = segment.origin.y - xy.y;
        let y0 = clamp(y, 0.0, 1.0);
        let y1 = clamp(y + segment.delta.y, 0.0, 1.0);
        let dy = y0 - y1;
        if dy != 0.0 {
            let vec_y_recip = 1.0 / segment.delta.y;
            let t0 = (y0 - y) * vec_y_recip;
            let t1 = (y1 - y) * vec_y_recip;
            let startx = segment.origin.x - xy.x;
            let x0 = startx + t0 * segment.delta.x;
            let x1 = startx + t1 * segment.delta.x;
            let xmin0 = min(x0, x1);
            let xmax0 = max(x0, x1);
            for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                let i_f = f32(i);
                let xmin = min(xmin0 - i_f, 1.0) - 1.0e-6;
                let xmax = xmax0 - i_f;
                let b = min(xmax, 1.0);
                let c = max(b, 0.0);
                let d = max(xmin, 0.0);
                let a = (b + 0.5 * (d * d - c * c) - xmin) / (xmax - xmin);
                area[i] += a * dy;
            }
        }
        let y_edge = sign(segment.delta.x) * clamp(xy.y - segment.y_edge + 1.0, 0.0, 1.0);
        for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
            area[i] += y_edge;
        }
        segment_ix = segment.next;
    }
    if even_odd {
        // even-odd winding rule
        for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
            let a = area[i];
            area[i] = abs(a - 2.0 * round(0.5 * a));
        }
    } else {
        // non-zero winding rule
        for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
            area[i] = min(abs(area[i]), 1.0);
        }
    }
    return area;
}

fn stroke_path(seg: u32, half_width: f32, xy: vec2<f32>) -> array<f32, PIXELS_PER_THREAD> {
    var df: array<f32, PIXELS_PER_THREAD>;
    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
        df[i] = 1e9;
    }
    var segment_ix = seg;
    while segment_ix != 0u {
        let segment = segments[segment_ix];
        let delta = segment.delta;
        let dpos0 = xy + vec2(0.5, 0.5) - segment.origin;
        let scale = 1.0 / dot(delta, delta);
        for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
            let dpos = vec2(dpos0.x + f32(i), dpos0.y);
            let t = clamp(dot(dpos, delta) * scale, 0.0, 1.0);
            // performance idea: hoist sqrt out of loop
            df[i] = min(df[i], length(delta * t - dpos));
        }
        segment_ix = segment.next;
    }
    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
        // reuse array; return alpha rather than distance
        df[i] = clamp(half_width + 0.5 - df[i], 0.0, 1.0);
    }
    return df;
}

// The X size should be 16 / PIXELS_PER_THREAD
@compute @workgroup_size(4, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let tile_ix = wg_id.y * config.width_in_tiles + wg_id.x;
    let xy = vec2(f32(global_id.x * PIXELS_PER_THREAD), f32(global_id.y));
#ifdef full
    var rgba: array<vec4<f32>, PIXELS_PER_THREAD>;
    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
        rgba[i] = unpack4x8unorm(config.base_color).wzyx;
    }
    var blend_stack: array<array<u32, PIXELS_PER_THREAD>, BLEND_STACK_SPLIT>;
    var clip_depth = 0u;
    var area: array<f32, PIXELS_PER_THREAD>;
    var cmd_ix = tile_ix * PTCL_INITIAL_ALLOC;
    let blend_offset = ptcl[cmd_ix];
    cmd_ix += 1u;
    // main interpretation loop
    while true {
        let tag = ptcl[cmd_ix];
        if tag == CMD_END {
            break;
        }
        switch tag {
            // CMD_FILL
            case 1u: {
                let fill = read_fill(cmd_ix);
                let segments = fill.tile >> 1u;
                let even_odd = (fill.tile & 1u) != 0u;
                let tile = Tile(fill.backdrop, segments);
                area = fill_path(tile, xy, even_odd);
                cmd_ix += 3u;
            }
            // CMD_STROKE
            case 2u: {
                let stroke = read_stroke(cmd_ix);
                area = stroke_path(stroke.tile, stroke.half_width, xy);
                cmd_ix += 3u;
            }
            // CMD_SOLID
            case 3u: {
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    area[i] = 1.0;
                }
                cmd_ix += 1u;
            }
            // CMD_COLOR
            case 5u: {
                let color = read_color(cmd_ix);
                let fg = unpack4x8unorm(color.rgba_color).wzyx;
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    let fg_i = fg * area[i];
                    rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                }
                cmd_ix += 2u;
            }
            // CMD_LIN_GRAD
            case 6u: {
                let lin = read_lin_grad(cmd_ix);
                let d = lin.line_x * xy.x + lin.line_y * xy.y + lin.line_c;
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    let my_d = d + lin.line_x * f32(i);
                    let x = i32(round(clamp(my_d, 0.0, 1.0) * f32(GRADIENT_WIDTH - 1)));
                    let fg_rgba = textureLoad(gradients, vec2(x, i32(lin.index)), 0);
                    let fg_i = fg_rgba * area[i];
                    rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                }
                cmd_ix += 3u;
            }
            // CMD_RAD_GRAD
            case 7u: {
                let rad = read_rad_grad(cmd_ix);
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    let my_xy = vec2(xy.x + f32(i), xy.y);
                    // TODO: can hoist y, but for now stick to the GLSL version
                    let xy_xformed = rad.matrx.xy * my_xy.x + rad.matrx.zw * my_xy.y + rad.xlat;
                    let ba = dot(xy_xformed, rad.c1);
                    let ca = rad.ra * dot(xy_xformed, xy_xformed);
                    let t = sqrt(ba * ba + ca) - ba - rad.roff;
                    let x = i32(round(clamp(t, 0.0, 1.0) * f32(GRADIENT_WIDTH - 1)));
                    let fg_rgba = textureLoad(gradients, vec2(x, i32(rad.index)), 0);
                    let fg_i = fg_rgba * area[i];
                    rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                }
                cmd_ix += 3u;
            }
            // CMD_IMAGE
            case 8u: {
                let image = read_image(cmd_ix);
                let atlas_extents = image.atlas_offset + image.extents;
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    let my_xy = vec2(xy.x + f32(i), xy.y);
                    let atlas_uv = image.matrx.xy * my_xy.x + image.matrx.zw * my_xy.y + image.xlat + image.atlas_offset;
                    // This currently clips to the image bounds. TODO: extend modes
                    if all(atlas_uv < atlas_extents) && area[i] != 0.0 {
                        let uv_quad = vec4(max(floor(atlas_uv), image.atlas_offset), min(ceil(atlas_uv), atlas_extents));
                        let uv_frac = fract(atlas_uv);
                        let a = premul_alpha(textureLoad(image_atlas, vec2<i32>(uv_quad.xy), 0));
                        let b = premul_alpha(textureLoad(image_atlas, vec2<i32>(uv_quad.xw), 0));
                        let c = premul_alpha(textureLoad(image_atlas, vec2<i32>(uv_quad.zy), 0));
                        let d = premul_alpha(textureLoad(image_atlas, vec2<i32>(uv_quad.zw), 0));
                        let fg_rgba = mix(mix(a, b, uv_frac.y), mix(c, d, uv_frac.y), uv_frac.x);
                        let fg_i = fg_rgba * area[i];
                        rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                    }
                }
                cmd_ix += 2u;
            }
            // CMD_BEGIN_CLIP
            case 9u: {
                if clip_depth < BLEND_STACK_SPLIT {
                    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                        blend_stack[clip_depth][i] = pack4x8unorm(rgba[i]);
                        rgba[i] = vec4(0.0);
                    }
                } else {
                    // TODO: spill to memory
                }
                clip_depth += 1u;
                cmd_ix += 1u;
            }
            // CMD_END_CLIP
            case 10u: {
                let end_clip = read_end_clip(cmd_ix);
                clip_depth -= 1u;
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    var bg_rgba: u32;
                    if clip_depth < BLEND_STACK_SPLIT {
                        bg_rgba = blend_stack[clip_depth][i];
                    } else {
                        // load from memory
                    }
                    let bg = unpack4x8unorm(bg_rgba);
                    let fg = rgba[i] * area[i] * end_clip.alpha;
                    rgba[i] = blend_mix_compose(bg, fg, end_clip.blend);
                }
                cmd_ix += 3u;
            }
            // CMD_JUMP
            case 11u: {
                cmd_ix = ptcl[cmd_ix + 1u];
            }
            default: {}
        }
    }
    let xy_uint = vec2<u32>(xy);
    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
        let coords = xy_uint + vec2(i, 0u);
        if coords.x < config.target_width && coords.y < config.target_height {
            let fg = rgba[i];
            // Max with a small epsilon to avoid NaNs
            let a_inv = 1.0 / max(fg.a, 1e-6);
            let rgba_sep = vec4(fg.rgb * a_inv, fg.a);            
            textureStore(output, vec2<i32>(coords), rgba_sep);
        }
    } 
#else
    let tile = tiles[tile_ix];
    let area = fill_path(tile, xy);

    let xy_uint = vec2<u32>(xy);
    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
        let coords = xy_uint + vec2(i, 0u);
        if coords.x < config.target_width && coords.y < config.target_height {
            textureStore(output, vec2<i32>(coords), vec4(area[i]));
        }
    }
#endif
}

fn premul_alpha(rgba: vec4<f32>) -> vec4<f32> {
    return vec4(rgba.rgb * rgba.a, rgba.a);
}
