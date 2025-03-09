// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// A simple render pipeline for solid color sparse strip rendering.
// Each instance draws one strip consisting of alpha values (dense_width) then a solid region.

struct VertexOutput {
    @location(0) tex_coord: vec2<f32>,
    @location(1) @interpolate(flat) dense_end: u32,
    @location(2) @interpolate(flat) color: u32,
    @builtin(position) position: vec4<f32>,
};

struct Config {
    width: u32,
    height: u32,
    strip_height: u32,
    strips_per_row: u32,
    alpha_texture_width: u32,
}

struct Strip {
    // [x, y] packed as u16's
    xy: u32,
    // [width, dense_width] packed as u16's
    widths: u32,
    col: u32,
    rgba: u32,
}

@group(0) @binding(1)
var<uniform> config: Config;

@group(0) @binding(2)
var strips_texture: texture_2d<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    @builtin(instance_index) in_instance_index: u32
) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(in_vertex_index & 1u);
    let y = f32(in_vertex_index >> 1u);
    
    // Read strip data from texture using 2D layout
    // Calculate texture coordinates based on instance index
    let strip_row = in_instance_index / config.strips_per_row;
    let strip_col = in_instance_index % config.strips_per_row;
    let base_x = strip_col * 4u; // Each strip takes 4 texels in a row
    
    // Read the 4 components of the strip from texture
    let xy = textureLoad(strips_texture, vec2<u32>(base_x, strip_row), 0).x;
    let widths = textureLoad(strips_texture, vec2<u32>(base_x + 1u, strip_row), 0).x;
    let col = textureLoad(strips_texture, vec2<u32>(base_x + 2u, strip_row), 0).x;
    let rgba = textureLoad(strips_texture, vec2<u32>(base_x + 3u, strip_row), 0).x;
    
    let x0 = xy & 0xffffu;
    let y0 = xy >> 16u;
    let width = widths & 0xffffu;
    let dense_width = widths >> 16u;
    out.dense_end = col + dense_width;
    let pix_x = f32(x0) + f32(width) * x;
    let pix_y = f32(y0) + y * f32(config.strip_height);
    let gl_x = (pix_x + 0.5) * 2.0 / f32(config.width) - 1.0;
    let gl_y = 1.0 - (pix_y + 0.5) * 2.0 / f32(config.height);
    out.position = vec4<f32>(gl_x, gl_y, 0.0, 1.0);
    out.tex_coord = vec2<f32>(f32(col) + x * f32(width), y * f32(config.strip_height));
    out.color = rgba;
    return out;
}

@group(0) @binding(0)
var alphas_texture: texture_2d<u32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let x = u32(floor(in.tex_coord.x));
    var alpha = 1.0;
    if x < in.dense_end {
        let y = u32(floor(in.tex_coord.y));
        // Read alpha value from texture
        // Calculate texture coordinates based on x for the u32 value
        let tex_x = x % config.alpha_texture_width;
        let tex_y = x / config.alpha_texture_width;
        let a = textureLoad(alphas_texture, vec2<u32>(tex_x, tex_y), 0).x;
        alpha = f32((a >> (y * 8u)) & 0xffu) * (1.0 / 255.0);
    }
    return alpha * unpack4x8unorm(in.color);
}
