// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// GPU winding accumulation for the hybrid fast path.
//
// Each instance renders a single tile-local contribution into a compact winding texture.
// The winding texture is laid out as a sequence of strip columns, with TILE_SIZE rows per
// column and wrapping in bands by texture width.

struct WindingConfig {
    winding_tex_width: u32,
    winding_tex_height: u32,
}

@group(0) @binding(0)
var<uniform> config: WindingConfig;

const TILE_SIZE = 4u;
const TILE_X_MASK = 0xffffu;
const TILE_Y_SHIFT = 16u;
const TILE_Y_MASK = 0x7fffu;
const KIND_BIT = 0x80000000u;
const NEARLY_ZERO = 1e-6;

struct TileLineInstance {
    @location(0) winding_col: u32,
    @location(1) tile_xy_kind: u32,
    @location(2) p0: vec2<f32>,
    @location(3) p1: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) p0: vec2<f32>,
    @location(1) p1: vec2<f32>,
    @location(2) @interpolate(flat) kind: u32,
    @location(3) @interpolate(flat) tile_origin: vec2<f32>,
    @location(4) screen_pos: vec2<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    instance: TileLineInstance,
) -> VertexOutput {
    let offset_x = f32(vertex_index & 1u) * f32(TILE_SIZE);
    let offset_y = f32(vertex_index >> 1u) * f32(TILE_SIZE);

    let packed = instance.tile_xy_kind;
    let tile_x = packed & TILE_X_MASK;
    let tile_y = (packed >> TILE_Y_SHIFT) & TILE_Y_MASK;
    let kind = packed & KIND_BIT;
    let tile_origin = vec2<f32>(f32(tile_x) * f32(TILE_SIZE), f32(tile_y) * f32(TILE_SIZE));

    let col = instance.winding_col + u32(offset_x);
    let band = col / config.winding_tex_width;
    let tex_x = col % config.winding_tex_width;
    let tex_y = band * TILE_SIZE + u32(offset_y);

    let ndc_x = f32(tex_x) * 2.0 / f32(config.winding_tex_width) - 1.0;
    let ndc_y = 1.0 - f32(tex_y) * 2.0 / f32(config.winding_tex_height);

    var clipped_p0 = instance.p0;
    var clipped_p1 = instance.p1;
    if kind == 0u {
        let tile_left = tile_origin.x;
        let tile_right = tile_left + f32(TILE_SIZE);
        let line_dx = clipped_p1.x - clipped_p0.x;

        if abs(line_dx) > NEARLY_ZERO {
            let t_left = (tile_left - clipped_p0.x) / line_dx;
            let t_right = (tile_right - clipped_p0.x) / line_dx;
            let t_enter = max(min(t_left, t_right), 0.0);
            let t_exit = min(max(t_left, t_right), 1.0);
            let original_p0 = clipped_p0;
            clipped_p0 = mix(original_p0, clipped_p1, t_enter);
            clipped_p1 = mix(original_p0, clipped_p1, t_exit);
        }
    }

    var out: VertexOutput;
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.p0 = clipped_p0;
    out.p1 = clipped_p1;
    out.kind = kind;
    out.tile_origin = tile_origin;
    out.screen_pos = tile_origin + vec2<f32>(offset_x, offset_y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if in.kind != 0u {
        let row = u32(floor(in.screen_pos.y - in.tile_origin.y));
        let windings = array<f32, 4>(in.p0.x, in.p0.y, in.p1.x, in.p1.y);
        return vec4<f32>(windings[row], 0.0, 0.0, 0.0);
    }

    let pixel_x = floor(in.screen_pos.x);
    let pixel_y = floor(in.screen_pos.y);

    let rel_p0 = in.p0 - vec2<f32>(pixel_x, pixel_y);
    let rel_p1 = in.p1 - vec2<f32>(pixel_x, pixel_y);

    let y0 = clamp(rel_p0.y, 0.0, 1.0);
    let y1 = clamp(rel_p1.y, 0.0, 1.0);
    let dy = y0 - y1;
    if abs(dy) < NEARLY_ZERO {
        return vec4<f32>(0.0);
    }

    let rel_dx = rel_p1.x - rel_p0.x;
    let rel_dy = rel_p1.y - rel_p0.y;
    let inv_slope = rel_dx / rel_dy;

    let x_at_y0 = rel_p0.x + (y0 - rel_p0.y) * inv_slope;
    let x_at_y1 = rel_p0.x + (y1 - rel_p0.y) * inv_slope;

    var xmin = min(x_at_y0, x_at_y1);
    let xmax = max(x_at_y0, x_at_y1);

    if (xmax - xmin) < NEARLY_ZERO {
        let coverage = clamp(1.0 - xmin, 0.0, 1.0);
        return vec4<f32>(coverage * dy, 0.0, 0.0, 0.0);
    }

    xmin = min(xmin, 1.0);
    let xmax_capped = min(xmax, 1.0);
    let x_hi = max(xmax_capped, 0.0);
    let x_lo = max(xmin, 0.0);
    let coverage = (xmax_capped + 0.5 * (x_lo * x_lo - x_hi * x_hi) - xmin) / (xmax - xmin);

    return vec4<f32>(coverage * dy, 0.0, 0.0, 0.0);
}
