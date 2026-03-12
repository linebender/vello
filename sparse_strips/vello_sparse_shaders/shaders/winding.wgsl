// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// Winding computation shader.
//
// Renders GpuTileLine instances into a compact winding texture. Each tile location
// occupies a TILE_SIZE × TILE_SIZE region in the texture, addressed by a column index.
//
// Two kinds of instances (distinguished by `kind`):
//   kind == 0: Analytic tile-line pair. Computes per-pixel winding from a line segment
//              clipped to the tile's x-bounds.
//   kind == 1: Coarse winding. p0/p1 encode per-row winding values to splat.
//
// Additive blending accumulates all contributions.

struct WindingConfig {
    winding_tex_width: u32,
    winding_tex_height: u32,
}

@group(0) @binding(0) var<uniform> config: WindingConfig;

const TILE_SIZE = 4u;
const TILE_X_MASK = 0xffffu;
const TILE_Y_SHIFT = 16u;
const TILE_Y_MASK = 0x7fffu;
const KIND_BIT = 0x80000000u;

struct TileLineInstance {
    @location(0) winding_col: u32,
    @location(1) tile_xy_kind: u32,
    @location(2) p0: vec2<f32>,
    @location(3) p1: vec2<f32>,
}

struct VsOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) p0: vec2<f32>,
    @location(1) p1: vec2<f32>,
    @location(2) @interpolate(flat) kind: u32,
    @location(3) @interpolate(flat) tile_origin: vec2<f32>,
    // Interpolated screen-space pixel position (for line-relative computation).
    @location(4) screen_pos: vec2<f32>,
}

const NEARLY_ZERO: f32 = 1e-6;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    instance: TileLineInstance,
) -> VsOutput {
    let ts = f32(TILE_SIZE);

    // Quad corner: (0,0), (1,0), (0,1), (1,1) scaled by tile size.
    let offset_x = f32(vertex_index & 1u) * ts;
    let offset_y = f32(vertex_index >> 1u) * ts;

    // Unpack tile position and kind.
    let packed = instance.tile_xy_kind;
    let tile_x = packed & TILE_X_MASK;
    let tile_y = (packed >> TILE_Y_SHIFT) & TILE_Y_MASK;
    let kind = packed & KIND_BIT;
    let tile_origin = vec2<f32>(f32(tile_x) * ts, f32(tile_y) * ts);

    // --- Position in the winding texture ---
    // Each tile location occupies TILE_SIZE adjacent columns, each TILE_SIZE pixels tall.
    let col = instance.winding_col + u32(offset_x);
    let band = col / config.winding_tex_width;
    let tex_x = col % config.winding_tex_width;
    let tex_y = band * TILE_SIZE + u32(offset_y);

    let ndc_x = f32(tex_x) * 2.0 / f32(config.winding_tex_width) - 1.0;
    let ndc_y = 1.0 - f32(tex_y) * 2.0 / f32(config.winding_tex_height);

    // --- Clip line to tile x-bounds (analytic only) ---
    var out_p0 = instance.p0;
    var out_p1 = instance.p1;

    if kind == 0u {
        let tile_left = tile_origin.x;
        let tile_right = tile_left + ts;
        let line_dx = out_p1.x - out_p0.x;

        if abs(line_dx) > NEARLY_ZERO {
            let t_left = (tile_left - out_p0.x) / line_dx;
            let t_right = (tile_right - out_p0.x) / line_dx;
            let t_enter = max(min(t_left, t_right), 0.0);
            let t_exit = min(max(t_left, t_right), 1.0);

            let orig_p0 = out_p0;
            out_p0 = mix(orig_p0, out_p1, t_enter);
            out_p1 = mix(orig_p0, out_p1, t_exit);
        }
    }

    var out: VsOutput;
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.p0 = out_p0;
    out.p1 = out_p1;
    out.kind = kind;
    out.tile_origin = tile_origin;
    out.screen_pos = tile_origin + vec2<f32>(offset_x, offset_y);
    return out;
}

@fragment
fn fs_main(in: VsOutput) -> @location(0) vec4<f32> {
    if in.kind != 0u {
        // Coarse: splat per-row winding value.
        let row = u32(floor(in.screen_pos.y - in.tile_origin.y));
        var windings = array<f32, 4>(in.p0.x, in.p0.y, in.p1.x, in.p1.y);
        return vec4<f32>(windings[row], 0.0, 0.0, 0.0);
    }

    let p0 = in.p0;
    let p1 = in.p1;

    let pixel_x = floor(in.screen_pos.x);
    let pixel_y = floor(in.screen_pos.y);

    let rel_p0 = p0 - vec2<f32>(pixel_x, pixel_y);
    let rel_p1 = p1 - vec2<f32>(pixel_x, pixel_y);

    let y0 = clamp(rel_p0.y, 0.0, 1.0);
    let y1 = clamp(rel_p1.y, 0.0, 1.0);
    let dy = y0 - y1;

    if abs(dy) < NEARLY_ZERO {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
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
