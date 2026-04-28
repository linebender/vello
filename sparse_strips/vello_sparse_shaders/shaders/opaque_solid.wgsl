// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

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

const RECT_STRIP_FLAG: u32 = 0x80000000u;

struct StripInstance {
    @location(0) xy: u32,
    @location(1) widths_or_rect_height: u32,
    @location(2) col_idx_or_rect_frac: u32,
    @location(3) payload: u32,
    @location(4) paint_and_rect_flag: u32,
    @location(5) depth_index: u32,
}

struct VertexOutput {
    @location(0) @interpolate(flat) color: vec4<f32>,
    @builtin(position) position: vec4<f32>,
}

@group(0) @binding(1)
var<uniform> config: Config;

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

    out.color = unpack4x8unorm(instance.payload);
    out.position = vec4<f32>(ndc_x, final_ndc_y, z, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
